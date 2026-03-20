"""
LLM integration with Ollama (local, CPU-friendly).
Uses phi3 for generation — runs on 8GB RAM.

Includes post-generation hallucination guard and confidence scoring.
"""
import re
import requests

MODEL_NAME = "phi3"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_CONTEXT_CHARS = 4000


def _ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    """Send prompt to Ollama and return response text."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 600,
            },
        }, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "Error: Empty response")
    except requests.ConnectionError:
        return "Error: Ollama not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error: LLM request failed — {e}"


def trim_context(chunks: list[dict]) -> list[dict]:
    """Select highest-value chunks that fit within the context budget."""
    ranked = sorted(
        chunks,
        key=lambda x: x.get("rerank_score", x.get("score", 0)),
        reverse=True,
    )

    seen_texts: set[str] = set()
    total = 0
    selected = []

    for c in ranked:
        fingerprint = c["text"][:200]
        if fingerprint in seen_texts:
            continue
        length = len(c["text"])
        if total + length > MAX_CONTEXT_CHARS:
            continue
        seen_texts.add(fingerprint)
        selected.append(c)
        total += length

    return selected


def ask(query: str, hits: list[dict]) -> dict:
    """Build numbered-context prompt, generate answer, then run hallucination guard.

    Returns dict with keys: answer, confidence, warnings
    """
    trimmed = trim_context(hits)

    MAX_CHUNK_LEN = 500
    context = "\n\n".join([
        f"[{i+1}] ({c['paper']} - {c['section']}): {c['text'][:MAX_CHUNK_LEN]}"
        for i, c in enumerate(trimmed[:5])
    ])

    sources = "\n".join([
        f"[{i+1}] {c['paper']} ({c['section']})"
        for i, c in enumerate(trimmed[:5])
    ])

    prompt = f"""You are a research assistant summarizing scientific papers about Alzheimer's disease.

Below are excerpts from research papers. Use them to answer the question.

{context}

Question: {query}

Instructions:
- Summarize what the sources say about this topic
- Write 3-5 bullet points, each starting with "- "
- After each claim, add the source number in brackets, e.g. [1], [2]
- Only use information from the sources above
- If asked about personal medical decisions, say "consult a healthcare professional"

Example format:
- Amyloid-beta plaques are a key hallmark of Alzheimer's disease [1].
- Tau protein tangles contribute to neuronal damage [2][3].
- Early detection using CSF biomarkers shows promise [1].

Answer:"""

    raw_answer = _ollama_generate(prompt, temperature=0.2)

    # Post-generation checks
    warnings = check_hallucination(raw_answer, trimmed)
    confidence = compute_confidence(hits, raw_answer, trimmed)

    # Flag low-confidence answers
    if confidence < 0.3:
        warnings.append("low_confidence")

    return {
        "answer": raw_answer,
        "confidence": confidence,
        "warnings": warnings,
    }


def check_hallucination(answer: str, chunks: list[dict]) -> list[str]:
    """Post-generation hallucination guard.

    Extracts key noun phrases from the answer and checks whether they
    appear in the source chunks.  Returns a list of warning strings
    for any claims that look unsupported.
    """
    if not chunks or not answer:
        return ["empty_context"]

    # Build a single lowercase reference corpus from all chunk texts
    corpus = " ".join(c["text"].lower() for c in chunks)

    # Pull out meaningful multi-word phrases (3+ word sequences)
    # that are likely factual claims worth verifying
    sentences = re.split(r"[.!?]", answer)
    warnings: list[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent.split()) < 5:
            continue

        # Extract key terms (words 4+ chars, not stopwords)
        key_terms = [
            w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", sent)
            if w.lower() not in _STOP_WORDS
        ]

        if not key_terms:
            continue

        # Check what fraction of key terms appear in the source corpus
        hits = sum(1 for t in key_terms if t in corpus)
        coverage = hits / len(key_terms) if key_terms else 0

        if coverage < 0.3:
            short = sent[:80] + ("..." if len(sent) > 80 else "")
            warnings.append(f"low_support: \"{short}\"")

    return warnings


def compute_confidence(hits: list[dict], answer: str, chunks_used: list[dict]) -> float:
    """Compute a 0-1 confidence score for the generated answer.

    Factors:
      - retrieval_score: avg rerank/cosine score of top chunks
      - citation_score:  how many [N] refs appear in the answer
      - overlap_score:   keyword overlap between answer and context
    """
    if not hits or not answer:
        return 0.0

    # 1. Retrieval quality (avg of top chunk scores, normalized to 0-1)
    scores = [h.get("rerank_score", h.get("score", 0)) for h in hits[:5]]
    retrieval_score = min(sum(scores) / max(len(scores), 1), 1.0)

    # 2. Citation density
    citations = re.findall(r"\[\d+\]", answer)
    citation_score = min(len(set(citations)) / 3.0, 1.0)  # 3+ unique refs = full marks

    # 3. Context overlap — what fraction of answer key terms appear in chunks
    corpus = " ".join(c["text"].lower() for c in chunks_used)
    answer_terms = [
        w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", answer)
        if w.lower() not in _STOP_WORDS
    ]
    if answer_terms:
        overlap_hits = sum(1 for t in answer_terms if t in corpus)
        overlap_score = overlap_hits / len(answer_terms)
    else:
        overlap_score = 0.0

    # Weighted combination
    confidence = (
        0.35 * retrieval_score
        + 0.30 * citation_score
        + 0.35 * overlap_score
    )
    return round(min(confidence, 1.0), 3)


# Common English stopwords to ignore in overlap checks
_STOP_WORDS = {
    "that", "this", "with", "from", "have", "been", "were", "also",
    "which", "their", "there", "they", "than", "these", "those",
    "into", "some", "such", "more", "most", "other", "about",
    "between", "through", "after", "before", "during", "each",
    "both", "does", "could", "would", "should", "being", "very",
    "when", "what", "where", "while", "only", "over", "under",
    "same", "well", "just", "like", "many", "much", "even",
}

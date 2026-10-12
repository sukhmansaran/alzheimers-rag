"""
LLM integration with Ollama (local, CPU-friendly).
Uses phi4-mini by default (override with OLLAMA_MODEL / OLLAMA_URL in .env) — runs on 8GB RAM.

Includes post-generation hallucination guard and confidence scoring.
"""
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "phi4-mini")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MAX_CONTEXT_CHARS = 4000
MAX_CHUNK_LEN = 500
MAX_CHUNKS = 5

DISCLAIMER = (
    "For research and educational use only. This summarizes research papers and is not "
    "medical advice; consult a healthcare professional about diagnosis or treatment."
)


class LLMError(RuntimeError):
    """Raised when the Ollama backend is unreachable or returns an error."""


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
        if resp.status_code == 404:
            raise LLMError(
                f"Model '{MODEL_NAME}' is not installed in Ollama — run `ollama pull {MODEL_NAME}` "
                "or set OLLAMA_MODEL in .env to a model you have (see `ollama list`)"
            )
        resp.raise_for_status()
    except requests.ConnectionError as e:
        raise LLMError("Ollama is not reachable — open the Ollama app or run `ollama serve`") from e
    except requests.RequestException as e:
        raise LLMError(f"LLM request failed — {e}") from e

    try:
        text = resp.json().get("response", "")
    except ValueError as e:
        raise LLMError("LLM returned a non-JSON response") from e
    if not text.strip():
        raise LLMError("LLM returned an empty response")
    return text


def trim_context(chunks: list[dict]) -> list[dict]:
    """Select highest-value chunks that fit within the context budget.

    Each chunk's text is truncated to MAX_CHUNK_LEN *before* budgeting, so the
    budget reflects what is actually sent to the model.
    """
    ranked = sorted(
        chunks,
        key=lambda x: x.get("rerank_score", x.get("score", 0)),
        reverse=True,
    )

    seen_texts: set[str] = set()
    total = 0
    selected = []

    for c in ranked:
        if len(selected) >= MAX_CHUNKS:
            break
        fingerprint = c["text"][:200]
        if fingerprint in seen_texts:
            continue
        text = c["text"][:MAX_CHUNK_LEN]
        if total + len(text) > MAX_CONTEXT_CHARS:
            continue
        seen_texts.add(fingerprint)
        selected.append({**c, "text": text})
        total += len(text)

    return selected


def ask(query: str, hits: list[dict]) -> dict:
    """Build numbered-context prompt, generate answer, then run hallucination guard.

    Returns dict with keys: answer, confidence, warnings, sources
    (sources[n-1] is the chunk cited as [n] in the answer).
    Raises LLMError if the Ollama backend fails.
    """
    trimmed = trim_context(hits)

    context = "\n\n".join([
        f"[{i+1}] ({c['paper']} - {c['section']}): {c['text']}"
        for i, c in enumerate(trimmed)
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
        "sources": trimmed,
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
      - retrieval_score: avg cosine similarity of top chunks (raw, not section-weighted)
      - citation_score:  how many [N] refs appear in the answer
      - overlap_score:   keyword overlap between answer and context
    """
    if not hits or not answer:
        return 0.0

    # 1. Retrieval quality (avg of top chunk scores, normalized to 0-1)
    scores = [h.get("score", 0.0) for h in hits[:MAX_CHUNKS]]
    retrieval_score = max(0.0, min(sum(scores) / max(len(scores), 1), 1.0))

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

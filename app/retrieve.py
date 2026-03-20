"""
Retrieval pipeline:
  Query → enhance → FAISS top-k → rerank → dedup (embedding) → diversify → final results.
"""
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.ingest import CHUNKS_PATH, EMBED_MODEL, INDEX_PATH

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Sections that typically carry the most signal
SECTION_WEIGHTS = {
    "Results": 1.3,
    "Discussion": 1.2,
    "Abstract": 1.2,   # dense factual summaries — boost
    "Conclusion": 1.1,
    "Methods": 1.0,
    "Methodology": 1.0,
    "Materials And Methods": 1.0,
    "Review": 1.1,
    "Limitations": 0.9,
    "Introduction": 0.85,
}

MIN_SCORE = 0.3
DEDUP_THRESHOLD = 0.95  # cosine similarity above this = duplicate

# Sections to exclude at retrieval time (safety net)
SKIP_SECTIONS = {"References", "Acknowledgments", "Preamble", "Keywords"}


def load_retriever():
    """Load model, index, and chunks once."""
    model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return model, index, chunks


def enhance_query(query: str) -> str:
    """Add domain context to improve embedding quality."""
    return f"Alzheimer's disease research: {query}"


def _faiss_search(query: str, model, index, chunks, k: int) -> list[dict]:
    """Raw FAISS cosine similarity search."""
    q_emb = model.encode([enhance_query(query)], convert_to_numpy=True)
    q_emb = q_emb.astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < MIN_SCORE:
            continue
        c = dict(chunks[idx])
        if c.get("section") in SKIP_SECTIONS:
            continue
        c["score"] = float(score)
        c["snippet"] = c["text"][:200]
        results.append(c)
    return results


def rerank(query: str, results: list[dict]) -> list[dict]:
    """Re-rank by combining cosine score + keyword overlap + section weight + chunk weight."""
    query_words = set(query.lower().split())

    for r in results:
        text_words = set(r["text"].lower().split())
        keyword_bonus = 0.01 * len(query_words & text_words)
        section_weight = SECTION_WEIGHTS.get(r["section"], 1.0)
        chunk_weight = r.get("weight", 1.0)
        r["rerank_score"] = (r["score"] + keyword_bonus) * section_weight * chunk_weight

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def _dedup_by_embedding(results: list[dict], model) -> list[dict]:
    """Remove semantically duplicate chunks using embedding cosine similarity."""
    if not results:
        return results

    texts = [r["text"] for r in results]
    embeddings = model.encode(texts, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)

    selected = [0]  # always keep the top result
    for i in range(1, len(results)):
        is_dup = False
        for j in selected:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            selected.append(i)

    return [results[i] for i in selected]


def diversify(results: list[dict], max_per_paper: int = 2) -> list[dict]:
    """Limit chunks per paper to ensure source diversity."""
    seen: dict[str, int] = {}
    final = []
    for r in results:
        p = r["paper"]
        if seen.get(p, 0) < max_per_paper:
            final.append(r)
            seen[p] = seen.get(p, 0) + 1
    return final


def _is_good_hit(text: str) -> bool:
    """Filter out low-quality hits post-retrieval."""
    return len(text.split()) > 50 and "et al" not in text.lower()


def search(query: str, model, index, chunks, k: int = 5) -> list[dict]:
    """Full pipeline: search → rerank → dedup (embedding) → diversify → quality filter → top-k."""
    raw = _faiss_search(query, model, index, chunks, k=k * 4)
    ranked = rerank(query, raw)
    deduped = _dedup_by_embedding(ranked, model)
    diverse = diversify(deduped)

    # Score threshold + quality filter
    filtered = [
        r for r in diverse
        if r["rerank_score"] >= MIN_SCORE and _is_good_hit(r["text"])
    ]

    for i, r in enumerate(filtered[:k]):
        r["rank"] = i + 1
    return filtered[:k]

"""
Retrieval pipeline (hybrid):
  Query → dense (FAISS) top-n + BM25 top-n → reciprocal-rank fusion × section weight
        → relevance + quality filter → dedup (embedding) → diversify → top-k.

Choices below were measured on app/eval_queries.json (python -m app.evaluate --retrieval-only)
and an off-topic probe set; see tests/eval_results.md.
"""
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.bm25 import BM25
from app.ingest import CHUNKS_PATH, EMBED_MODEL, INDEX_META_PATH, INDEX_PATH, QUERY_INSTRUCTION

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

# Cosine similarity floor (bge-small scale). Measured: weakest on-topic eval question's best
# chunk = 0.644; strongest off-topic probe ("capital of France", "bake bread", ...) = 0.500.
MIN_SCORE = 0.55
CANDIDATES = 20   # taken from each of dense and BM25 before fusion
RRF_K = 60        # standard reciprocal-rank-fusion constant
DEDUP_THRESHOLD = 0.95  # cosine similarity above this = duplicate

# Sections to exclude at retrieval time (safety net)
SKIP_SECTIONS = {"References", "Acknowledgments", "Preamble", "Keywords"}


def load_retriever():
    """Load model, index, and chunks once."""
    for path in (INDEX_PATH, CHUNKS_PATH, INDEX_META_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run `python -m app.ingest` first")

    with open(INDEX_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("embed_model") != EMBED_MODEL:
        # Same vector size across models, so a mismatch would silently return wrong results.
        raise RuntimeError(
            f"Index was built with {meta.get('embed_model')!r} but the app uses {EMBED_MODEL!r} — "
            "re-run `python -m app.ingest`"
        )

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index has {index.ntotal} vectors but chunks.json has {len(chunks)} chunks — "
            "re-run `python -m app.ingest`"
        )

    model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    return model, index, chunks


_bm25_cache: tuple[list[dict], BM25] | None = None


def _bm25_for(chunks: list[dict]) -> BM25:
    """Build the BM25 index once per chunk list (cheap: hundreds of chunks)."""
    global _bm25_cache
    if _bm25_cache is None or _bm25_cache[0] is not chunks:
        _bm25_cache = (chunks, BM25([c["text"] for c in chunks]))
    return _bm25_cache[1]


def _encode_query(query: str, model) -> np.ndarray:
    q = model.encode([QUERY_INSTRUCTION + query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)
    return q


def _hybrid_candidates(query: str, model, index, chunks) -> list[dict]:
    """Union of dense and BM25 top candidates, each with cosine score and fused RRF score."""
    q = _encode_query(query, model)
    _, dense_ids = index.search(q, min(CANDIDATES, index.ntotal))
    dense_rank = [int(i) for i in dense_ids[0] if i >= 0]

    bm = _bm25_for(chunks).scores(query)
    bm_rank = [i for i in np.argsort(bm)[::-1][:CANDIDATES] if bm[i] > 0]

    fused: dict[int, float] = {}
    for ranking in (dense_rank, bm_rank):
        for r, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + 1 / (RRF_K + r + 1)

    results = []
    for i, rrf in fused.items():
        c = dict(chunks[i])
        if c.get("section") in SKIP_SECTIONS:
            continue
        c["score"] = float(index.reconstruct(i) @ q[0])  # cosine similarity (vectors are normalized)
        c["rrf"] = rrf
        c["snippet"] = c["text"][:200]
        c["index_id"] = i  # row in the FAISS index, reused by dedup
        results.append(c)
    return results


def rerank(results: list[dict]) -> list[dict]:
    """Order by fused rank score × section weight, and attach a 0-1 relevance from cosine.

    Only SECTION_WEIGHTS is applied. The ingest-time chunk "weight" also encodes
    section, so multiplying both double-counted it (Results ~2.6x).
    """
    for r in results:
        r["rerank_score"] = r["rrf"] * SECTION_WEIGHTS.get(r["section"], 1.0)
        # Rescale similarity so MIN_SCORE → 0 and 1.0 → 1 (used by confidence scoring)
        r["relevance"] = max(0.0, min((r["score"] - MIN_SCORE) / (1 - MIN_SCORE), 1.0))
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def _dedup_by_embedding(results: list[dict], index) -> list[dict]:
    """Remove semantically duplicate chunks using embedding cosine similarity.

    Reuses the (already normalized) vectors stored in the FAISS index instead of
    re-encoding every candidate on each query.
    """
    if not results:
        return results

    embeddings = np.stack([index.reconstruct(r["index_id"]) for r in results])

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


MAX_ET_AL = 2  # more citations than this suggests a reference-list fragment


def _is_good_hit(text: str) -> bool:
    """Filter out low-quality hits post-retrieval.

    A chunk that merely cites prior work ("the Velasco et al. protocol") is kept;
    only citation-dense chunks are dropped.
    """
    return len(text.split()) > 50 and text.lower().count("et al") <= MAX_ET_AL


def search(query: str, model, index, chunks, k: int = 5) -> list[dict]:
    """Full pipeline: hybrid candidates → rerank → relevance/quality filter → dedup → diversify → top-k.

    Filtering runs before dedup/diversify so a low-quality chunk can't claim a
    dedup or per-paper slot and then be discarded, taking good chunks with it.
    Returns [] when nothing clears MIN_SCORE (e.g. off-topic questions).
    """
    ranked = rerank(_hybrid_candidates(query, model, index, chunks))

    filtered = [r for r in ranked if r["score"] >= MIN_SCORE and _is_good_hit(r["text"])]

    deduped = _dedup_by_embedding(filtered, index)
    diverse = diversify(deduped)

    for i, r in enumerate(diverse[:k]):
        r["rank"] = i + 1
    return diverse[:k]

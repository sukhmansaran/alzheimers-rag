"""Retrieval pipeline tests using a real FAISS index and a deterministic fake embedder."""
import json

import faiss
import numpy as np
import pytest

from app import retrieve
from app.retrieve import _is_good_hit, diversify, rerank, search


class FakeModel:
    """Embeds text by keyword: each topic word maps to its own axis."""

    TOPICS = ["amyloid", "tau", "apoe", "sleep"]

    def encode(self, texts, convert_to_numpy=True, **_):
        vecs = np.full((len(texts), len(self.TOPICS)), 0.05, dtype=np.float32)
        for i, t in enumerate(texts):
            for j, topic in enumerate(self.TOPICS):
                if topic in t.lower():
                    vecs[i, j] = 1.0
        return vecs


def _chunk(text_topic: str, paper: str, section: str = "Results", n_words: int = 60) -> dict:
    return {"text": f"{text_topic} " + " ".join(f"w{i}" for i in range(n_words)),
            "paper": paper, "section": section, "weight": 1.0}


def _build(chunks):
    model = FakeModel()
    emb = model.encode([c["text"] for c in chunks])
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index


def test_diversify_caps_per_paper():
    results = [{"paper": p} for p in ["a", "a", "a", "b", "a", "c"]]
    assert [r["paper"] for r in diversify(results)] == ["a", "a", "b", "c"]


def test_rerank_applies_section_weight():
    results = [
        {"text": "x", "score": 0.6, "section": "Introduction"},
        {"text": "x", "score": 0.55, "section": "Results"},
    ]
    assert [r["section"] for r in rerank("q", results)] == ["Results", "Introduction"]


def test_search_returns_relevant_ranked_hits():
    chunks = [_chunk("amyloid", "a.pdf"), _chunk("tau", "b.pdf"), _chunk("amyloid tau", "c.pdf"),
              _chunk("sleep", "d.pdf")]
    model, index = _build(chunks)

    hits = search("amyloid plaques", model, index, chunks, k=3)

    assert hits, "expected at least one hit"
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    assert all("amyloid" in h["text"] for h in hits)
    assert "d.pdf" not in {h["paper"] for h in hits}


def test_search_drops_duplicates_and_short_or_citation_chunks():
    chunks = [
        _chunk("amyloid", "a.pdf"),
        _chunk("amyloid", "b.pdf"),                        # semantic duplicate of the first
        _chunk("amyloid", "c.pdf", n_words=10),            # too short
        {**_chunk("amyloid", "d.pdf"), "text": "amyloid " + "Smith et al, " * 3 + " ".join(["w"] * 60)},
    ]
    model, index = _build(chunks)

    hits = search("amyloid", model, index, chunks, k=5)

    assert len(hits) == 1


def test_load_retriever_rejects_mismatched_index_and_chunks(tmp_path, monkeypatch):
    chunks = [_chunk("amyloid", "a.pdf"), _chunk("tau", "b.pdf")]
    _, index = _build(chunks)
    faiss.write_index(index, str(tmp_path / "e.faiss"))
    (tmp_path / "c.json").write_text(json.dumps(chunks[:1]))
    monkeypatch.setattr(retrieve, "INDEX_PATH", str(tmp_path / "e.faiss"))
    monkeypatch.setattr(retrieve, "CHUNKS_PATH", str(tmp_path / "c.json"))

    with pytest.raises(RuntimeError, match="re-run"):
        retrieve.load_retriever()


def test_load_retriever_missing_files_says_to_ingest(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "INDEX_PATH", str(tmp_path / "missing.faiss"))
    with pytest.raises(FileNotFoundError, match="app.ingest"):
        retrieve.load_retriever()


def test_single_citation_is_kept_but_citation_dense_chunk_is_dropped():
    body = " ".join(["w"] * 60)
    assert _is_good_hit(f"We adopted the Velasco et al. protocol. {body}")
    assert not _is_good_hit(f"Smith et al. Jones et al. Lee et al. {body}")

"""Unit tests for context trimming in app.llm (no Ollama required)."""
import pytest

from app import llm
from app.llm import MAX_CHUNK_LEN, MAX_CHUNKS, MAX_CONTEXT_CHARS, LLMError, compute_confidence, trim_context


def _chunk(i: int, length: int = 3000, score: float = 0.5) -> dict:
    return {"text": f"chunk{i} " + "x" * length, "paper": f"p{i}.pdf", "section": "Results",
            "rerank_score": score}


def test_long_chunks_are_truncated_before_budgeting():
    # Regression: 3000-char chunks used to exhaust the 4000-char budget after one chunk.
    selected = trim_context([_chunk(i) for i in range(5)])
    assert len(selected) == 5
    assert all(len(c["text"]) <= MAX_CHUNK_LEN for c in selected)
    assert sum(len(c["text"]) for c in selected) <= MAX_CONTEXT_CHARS


def test_caps_chunk_count_and_orders_by_score():
    chunks = [_chunk(i, score=i / 10) for i in range(10)]
    selected = trim_context(chunks)
    assert len(selected) == MAX_CHUNKS
    assert selected[0]["paper"] == "p9.pdf"


def test_duplicates_are_dropped():
    c = _chunk(1)
    assert len(trim_context([c, dict(c)])) == 1


def test_input_chunks_are_not_mutated():
    c = _chunk(1)
    original = c["text"]
    trim_context([c])
    assert c["text"] == original


def test_confidence_uses_raw_similarity_not_weighted_score():
    # Weak cosine matches inflated by section weights must not look like strong retrieval.
    hits = [{"score": 0.3, "rerank_score": 0.3 * 1.3 * 2.0, "text": "x"}] * 5
    confidence = compute_confidence(hits, "unrelated words entirely", [{"text": "abc"}])
    assert confidence < 0.3


def test_non_json_ollama_response_raises_llm_error(monkeypatch):
    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: Resp())
    with pytest.raises(LLMError, match="non-JSON"):
        llm._ollama_generate("prompt")


def test_ask_returns_the_sources_it_cited(monkeypatch):
    monkeypatch.setattr(llm, "_ollama_generate", lambda prompt, temperature=0.2: "- Claim [1].")
    hits = [_chunk(i, score=s) for i, s in enumerate([0.2, 0.9, 0.5])]
    result = llm.ask("q", hits)
    assert [c["paper"] for c in result["sources"]] == ["p1.pdf", "p2.pdf", "p0.pdf"]


def test_missing_model_gives_actionable_error(monkeypatch):
    class Resp:
        status_code = 404

    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: Resp())
    with pytest.raises(LLMError, match="ollama pull"):
        llm._ollama_generate("prompt")

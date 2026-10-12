"""Unit tests for context trimming in app.llm (no Ollama required)."""
from app.llm import MAX_CHUNK_LEN, MAX_CHUNKS, MAX_CONTEXT_CHARS, trim_context


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

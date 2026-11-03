"""API tests with the retriever and LLM stubbed out (no model, index, or Ollama needed)."""
import pytest
from fastapi.testclient import TestClient

from app import api
from app.llm import LLMError

HIT = {
    "rank": 1, "paper": "paper.pdf", "section": "Results", "score": 0.81234,
    "text": "Plasma p-tau217 predicted amyloid positivity.", "snippet": "Plasma p-tau217",
}


@pytest.fixture
def client(monkeypatch):
    # Without `with TestClient(...)` the lifespan (which loads the real model) does not run.
    monkeypatch.setattr(api, "_model", object())
    monkeypatch.setattr(api, "_index", object())
    monkeypatch.setattr(api, "_chunks", [{"paper": "b.pdf"}, {"paper": "a.pdf"}, {"paper": "a.pdf"}])
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks_loaded": 3}


def test_sources_are_unique_and_sorted(client):
    assert client.get("/sources").json() == {"papers": ["a.pdf", "b.pdf"], "total_chunks": 3}


def test_query_success(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [dict(HIT)])
    monkeypatch.setattr(api, "ask", lambda q, hits: {
        "answer": "- Finding [1].", "confidence": 0.7, "warnings": [], "sources": hits[:1],
    })

    body = client.post("/query", json={"question": "biomarkers?"}).json()
    assert body["answer"] == "- Finding [1]."
    assert body["confidence"] == 0.7
    assert body["sources"][0] == {
        "rank": 1, "paper": "paper.pdf", "section": "Results", "score": 0.8123, "snippet": "Plasma p-tau217",
        "excerpt": "Plasma p-tau217 predicted amyloid positivity.",
    }


def test_sources_are_the_chunks_the_llm_saw(client, monkeypatch):
    hits = [dict(HIT, paper=f"p{i}.pdf") for i in range(8)]
    monkeypatch.setattr(api, "search", lambda *a, **kw: hits)
    monkeypatch.setattr(api, "ask", lambda q, h: {
        "answer": "x [1][2]", "confidence": 0.5, "warnings": [], "sources": [h[3], h[1]],
    })
    sources = client.post("/query", json={"question": "q", "k": 8}).json()["sources"]
    assert [(s["rank"], s["paper"]) for s in sources] == [(1, "p3.pdf"), (2, "p1.pdf")]


def test_query_no_hits_returns_valid_response(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [])
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 200
    assert resp.json()["sources"] == []
    assert resp.json()["warnings"] == ["no_sources"]


def test_query_llm_failure_returns_503(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [dict(HIT)])

    def fail(q, hits):
        raise LLMError("Ollama not running")

    monkeypatch.setattr(api, "ask", fail)
    resp = client.post("/query", json={"question": "biomarkers?"})
    assert resp.status_code == 503
    assert "Ollama" in resp.json()["detail"]


@pytest.mark.parametrize("payload, status", [
    ({"question": "   "}, 400),
    ({"question": "q", "k": 0}, 422),
    ({"question": "q", "k": 21}, 422),
])
def test_query_validation(client, payload, status):
    assert client.post("/query", json=payload).status_code == status


def test_query_when_retriever_not_loaded(monkeypatch):
    monkeypatch.setattr(api, "_chunks", None)
    assert TestClient(api.app).post("/query", json={"question": "q"}).status_code == 503


def test_query_includes_disclaimer(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [])
    assert "not medical advice" in client.post("/query", json={"question": "q"}).json()["disclaimer"]


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    monkeypatch.setattr(api, "search", lambda *a, **kw: [])

    assert client.post("/query", json={"question": "q"}).status_code == 401
    assert client.post("/query", json={"question": "q"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/query", json={"question": "q"}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/sources").status_code == 401
    assert client.get("/health").status_code == 200  # health stays open for monitoring


def test_api_key_not_required_when_unset(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert client.get("/sources").status_code == 200


def test_cors_allows_localhost_but_not_other_origins(client):
    def preflight(origin):
        return client.options("/query", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
        }).headers.get("access-control-allow-origin")

    assert preflight("http://localhost:3000") == "http://localhost:3000"
    assert preflight("https://evil.example.com") is None


def _events(resp):
    import json
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_query_stream_sends_sources_tokens_then_done(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [dict(HIT, rerank_score=1.0)])
    monkeypatch.setattr(api, "stream_generate", lambda prompt: iter(["- Plasma p-tau217 ", "predicted amyloid [1]."]))

    resp = client.post("/query/stream", json={"question": "biomarkers?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _events(resp)

    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[0]["sources"][0]["paper"] == "paper.pdf"
    assert events[0]["sources"][0]["excerpt"] == HIT["text"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "- Plasma p-tau217 predicted amyloid [1]."
    assert 0 <= events[-1]["confidence"] <= 1
    assert "not medical advice" in events[-1]["disclaimer"]


def test_query_stream_reports_llm_failure_as_error_event(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [dict(HIT, rerank_score=1.0)])

    def fail(prompt):
        raise LLMError("Ollama is not reachable")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(api, "stream_generate", fail)
    events = _events(client.post("/query/stream", json={"question": "q"}))
    assert [e["type"] for e in events] == ["sources", "error"]
    assert "Ollama" in events[-1]["detail"]


def test_query_stream_no_sources(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **kw: [])
    events = _events(client.post("/query/stream", json={"question": "capital of France?"}))
    assert [e["type"] for e in events] == ["sources", "token", "done"]
    assert events[-1]["warnings"] == ["no_sources"]


def test_query_stream_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.post("/query/stream", json={"question": "q"}).status_code == 401


def test_web_ui_is_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "/query/stream" in resp.text
    # Untrusted text (papers, LLM output) must never be injected as HTML
    assert "innerHTML" not in resp.text

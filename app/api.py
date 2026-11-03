"""
FastAPI server for the Alzheimer's RAG pipeline.

Usage:
    uvicorn app.api:app --reload --port 8000

Security settings (.env):
    API_KEY       If set, /query and /sources require header `X-API-Key: <API_KEY>`.
    CORS_ORIGINS  Comma-separated allowed browser origins. Default: localhost only.
"""
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.llm import DISCLAIMER, LLMError, ask, build_prompt, finalize_answer, stream_generate
from app.retrieve import load_retriever, search

# ── Global state ──────────────────────────────────────────────

_model = None
_index = None
_chunks = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retriever once at startup."""
    global _model, _index, _chunks
    print("Loading retriever...")
    _model, _index, _chunks = load_retriever()
    print(f"Ready — {len(_chunks)} chunks indexed")
    yield


app = FastAPI(
    title="Alzheimer's RAG API",
    description="Grounded research assistant for Alzheimer's disease literature",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Without CORS_ORIGINS, only local pages (any port) may call the API from a browser.
    allow_origin_regex=None if _cors_origins else r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Enforce X-API-Key when API_KEY is configured; open access otherwise (local dev)."""
    expected = os.environ.get("API_KEY")
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: int = Field(5, ge=1, le=20)


class Source(BaseModel):
    rank: int
    paper: str
    section: str
    score: float
    snippet: str
    excerpt: str  # exact text the LLM was given for this source


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    warnings: list[str]
    sources: list[Source]
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _chunks else "not_ready",
        chunks_loaded=len(_chunks) if _chunks else 0,
    )


NO_SOURCES_ANSWER = "No relevant sources found for this question."


def _retrieve(req: QueryRequest) -> list[dict]:
    if _model is None or _index is None or not _chunks:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return search(req.question, _model, _index, _chunks, k=req.k)


def _to_sources(chunks_used: list[dict]) -> list[Source]:
    """Numbered to match the [n] citations in the answer."""
    return [
        Source(
            rank=i + 1,
            paper=h["paper"],
            section=h["section"],
            score=round(h["score"], 4),
            snippet=h.get("snippet", h["text"][:200]),
            excerpt=h["text"],
        )
        for i, h in enumerate(chunks_used)
    ]


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    hits = _retrieve(req)

    if not hits:
        return QueryResponse(
            question=req.question,
            answer=NO_SOURCES_ANSWER,
            confidence=0.0,
            warnings=["no_sources"],
            sources=[],
        )

    # Generate answer (returns dict with answer, confidence, warnings)
    try:
        result = ask(req.question, hits)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        confidence=result["confidence"],
        warnings=result["warnings"],
        sources=_to_sources(result["sources"]),
    )


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
def query_stream(req: QueryRequest):
    """Stream the answer as newline-delimited JSON events:

    {"type": "sources", "sources": [...]}            once, before generation
    {"type": "token", "text": "..."}                 repeated
    {"type": "done", "confidence", "warnings", "disclaimer"}
    {"type": "error", "detail": "..."}               instead of "done" if the LLM fails
    """
    hits = _retrieve(req)

    def line(event: dict) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    def events():
        if not hits:
            yield line({"type": "sources", "sources": []})
            yield line({"type": "token", "text": NO_SOURCES_ANSWER})
            yield line({"type": "done", "confidence": 0.0, "warnings": ["no_sources"], "disclaimer": DISCLAIMER})
            return

        prompt, used = build_prompt(req.question, hits)
        yield line({"type": "sources", "sources": [s.model_dump() for s in _to_sources(used)]})

        pieces = []
        try:
            for piece in stream_generate(prompt):
                pieces.append(piece)
                yield line({"type": "token", "text": piece})
        except LLMError as e:
            yield line({"type": "error", "detail": str(e)})
            return

        result = finalize_answer("".join(pieces), hits, used)
        yield line({
            "type": "done",
            "confidence": result["confidence"],
            "warnings": result["warnings"],
            "disclaimer": DISCLAIMER,
        })

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/sources", dependencies=[Depends(require_api_key)])
def list_sources():
    """List all unique papers in the index."""
    if not _chunks:
        return {"papers": []}
    papers = sorted(set(c["paper"] for c in _chunks))
    return {"papers": papers, "total_chunks": len(_chunks)}

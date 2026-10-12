"""
FastAPI server for the Alzheimer's RAG pipeline.

Usage:
    uvicorn app.api:app --reload --port 8000

Security settings (.env):
    API_KEY       If set, /query and /sources require header `X-API-Key: <API_KEY>`.
    CORS_ORIGINS  Comma-separated allowed browser origins. Default: localhost only.
"""
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.llm import DISCLAIMER, LLMError, ask
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


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    if _model is None or _index is None or not _chunks:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve
    hits = search(req.question, _model, _index, _chunks, k=req.k)

    if not hits:
        return QueryResponse(
            question=req.question,
            answer="No relevant sources found for this question.",
            confidence=0.0,
            warnings=["no_sources"],
            sources=[],
        )

    # Generate answer (returns dict with answer, confidence, warnings)
    try:
        result = ask(req.question, hits)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Numbered to match the [n] citations in the answer
    sources = [
        Source(
            rank=i + 1,
            paper=h["paper"],
            section=h["section"],
            score=round(h["score"], 4),
            snippet=h.get("snippet", h["text"][:200]),
        )
        for i, h in enumerate(result["sources"])
    ]

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        confidence=result["confidence"],
        warnings=result["warnings"],
        sources=sources,
    )


@app.get("/sources", dependencies=[Depends(require_api_key)])
def list_sources():
    """List all unique papers in the index."""
    if not _chunks:
        return {"papers": []}
    papers = sorted(set(c["paper"] for c in _chunks))
    return {"papers": papers, "total_chunks": len(_chunks)}

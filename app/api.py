"""
FastAPI server for the Alzheimer's RAG pipeline.

Usage:
    uvicorn app.api:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.llm import ask
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: int = 5


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


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _chunks else "not_ready",
        chunks_loaded=len(_chunks) if _chunks else 0,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not _model or not _index or not _chunks:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve
    hits = search(req.question, _model, _index, _chunks, k=req.k)

    if not hits:
        return QueryResponse(
            question=req.question,
            answer="No relevant sources found for this question.",
            sources=[],
        )

    # Generate answer (returns dict with answer, confidence, warnings)
    result = ask(req.question, hits)

    sources = [
        Source(
            rank=h.get("rank", i + 1),
            paper=h["paper"],
            section=h["section"],
            score=round(h["score"], 4),
            snippet=h.get("snippet", h["text"][:200]),
        )
        for i, h in enumerate(hits)
    ]

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        confidence=result["confidence"],
        warnings=result["warnings"],
        sources=sources,
    )


@app.get("/sources")
async def list_sources():
    """List all unique papers in the index."""
    if not _chunks:
        return {"papers": []}
    papers = sorted(set(c["paper"] for c in _chunks))
    return {"papers": papers, "total_chunks": len(_chunks)}

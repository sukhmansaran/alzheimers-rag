# Alzheimer's RAG Research Assistant

[![CI](https://github.com/sukhmansaran/alzheimers-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/sukhmansaran/alzheimers-rag/actions/workflows/ci.yml)

A domain-specific, grounded research assistant for Alzheimer's disease literature. Built with section-aware chunking, FAISS retrieval, reranking, and local LLM inference via Ollama (phi4-mini by default).

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- ~8GB RAM (CPU-only, no GPU required)

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
# (for development: pip install -r requirements-dev.txt)

# 3. Pull the model
ollama pull phi4-mini
```

Ollama must be running. The desktop app starts it in the background automatically;
without the app, run `ollama serve` in a separate terminal.

To use a different model or Ollama host, set these in `.env`:

```
OLLAMA_MODEL=llama3.2:3b
OLLAMA_URL=http://localhost:11434/api/generate
```

## How It Works

```
PDF files (data/)
    │
    ▼
┌──────────────────┐
│    Ingestion      │  python -m app.ingest
│                   │
│  PDF → text       │
│  → sections       │  (academic heading detection + fallback)
│  → weighted chunks│  (section-aware, 400-word base)
│  → quality filter │  (noise/reference removal)
│  → embeddings     │  (BAAI/bge-small-en-v1.5)
│  → FAISS index    │
└────────┬─────────┘
         │ saved to processed/
         ▼
┌──────────────────┐
│    Retrieval      │  query → dense (FAISS) + keyword (BM25)
│                   │  → rank fusion × section weight
│                   │  → relevance floor (off-topic → no sources)
│                   │  → embedding dedup
│                   │  → diversify (max 2 per paper)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM (Ollama)     │  context trimming (4k char budget)
│  phi4-mini        │  → structured prompt (cited bullet points)
│                   │  → hallucination guard
│                   │  → confidence scoring
└──────────────────┘
```

## Usage

```bash
# Step 1: Ingest PDFs (run once, or when you add new papers)
python -m app.ingest

# Step 2: Ask questions via CLI
python -m app.main "What biomarkers are used for early detection of Alzheimer's?"

# Step 3: Run evaluation suite
python -m app.evaluate

# Step 4: Start the API server + web UI, then open http://localhost:8000
uvicorn app.api:app --reload --port 8000

# Tests + lint (no Ollama, PDFs, or index required)
pytest
ruff check app tests
```

## Web UI

Open http://localhost:8000 once the server is running. Answers stream in as they are generated; each `[n]`
citation is clickable and highlights the exact excerpt the model was given, so you can check every claim.
Confidence, guard warnings, and the disclaimer are shown under the answer. Off-topic questions get
"no relevant sources" rather than a made-up answer.

## API Endpoints

Once the server is running:

| Method | Endpoint   | Description                    |
|--------|-----------|--------------------------------|
| GET    | /         | Web UI                         |
| GET    | /health   | Check if retriever is loaded   |
| POST   | /query    | Ask a question (JSON body)     |
| POST   | /query/stream | Same, streamed as newline-delimited JSON (sources, tokens, done) |
| GET    | /sources  | List all indexed papers        |
| GET    | /docs     | Interactive Swagger UI         |

Example query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the risk factors for Alzheimer?"}'
```

Every `/query` response includes a `disclaimer` field (research use only, not medical advice); the CLI prints it too.

### Security (before exposing the API beyond your machine)

Set these in `.env`:

```
API_KEY=some-long-random-string              # /query and /sources then require header X-API-Key
CORS_ORIGINS=https://your-frontend.example.com  # comma-separated; default allows localhost only
```

With no `API_KEY`, the API is open, which is fine for local use. `/health` is always open.

## Evaluation

`python -m app.evaluate` runs four checks (`--retrieval-only` runs just the first, no Ollama needed):

- Labelled retrieval — 44 questions in [app/eval_queries.json](app/eval_queries.json), each tagged with the paper(s) that answer it; reports hit@5 and MRR, split into "easy" (the paper's own terms) and "hard" (paraphrased in plain language)
- Retrieval sanity — source diversity, score thresholds
- Answers — grounding (sentence-level key-term coverage of the context the LLM saw), valid citations, answer length
- Safety — refuses medical advice, no fabricated claims

## CI

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every push to `master` and on pull requests:

- **Lint**: `ruff check`
- **Test**: `pytest` on Python 3.10 and 3.12. The embedding model and Ollama are stubbed, so CI needs no PDFs, index, or LLM.

The full `python -m app.evaluate` suite needs Ollama and the ingested corpus, so run it locally.

## Project Structure

```
├── data/               # Source PDFs (not tracked)
├── processed/          # chunks.json + embeddings.faiss (generated)
├── app/
│   ├── ingest.py       # PDF → chunks → embeddings → FAISS index
│   ├── retrieve.py     # Hybrid search → fuse → filter → dedup → diversify
│   ├── bm25.py         # BM25 keyword scoring
│   ├── eval_queries.json # Labelled eval questions
│   ├── llm.py          # Ollama integration, prompt, guards
│   ├── evaluate.py     # Labelled retrieval + answer + safety evaluation
│   ├── api.py          # FastAPI server (+ serves the web UI)
│   ├── static/index.html # Web UI (single page, no build step)
│   ├── main.py         # CLI entry point
│   └── utils.py        # Text cleaning, section detection, chunking
├── tests/
│   ├── audit_data.py   # Data audit utility
│   ├── test_utils.py   # Unit tests: cleaning, sections, chunking
│   ├── test_llm.py     # Unit tests: context trimming
│   ├── test_retrieve.py# Retrieval pipeline (real FAISS, fake embedder)
│   ├── test_api.py     # API endpoints (retriever + LLM stubbed)
│   └── eval_results.md # Evaluation results log
├── .env                # HF_TOKEN, OLLAMA_MODEL, OLLAMA_URL, API_KEY, CORS_ORIGINS (not tracked)
├── .github/workflows/ci.yml  # Lint + test pipeline
├── requirements.txt         # exact pinned versions
├── requirements-dev.txt     # pytest, httpx, ruff (pinned)
├── pyproject.toml           # pytest + ruff config
└── .gitignore
```

## Design Decisions

- Section-aware chunking preserves academic structure (Abstract, Methods, Results, etc.)
- Hybrid retrieval: dense + BM25 merged with reciprocal-rank fusion. Exact terms (APOE ε4, p-tau217) come from BM25, paraphrases from embeddings
- Relevance floor (cosine 0.55) sits between the weakest on-topic (0.644) and strongest off-topic (0.500) score measured, so off-topic questions get no sources instead of a forced answer
- Section weighting at rerank time (applied once): Results 1.3 > Discussion/Abstract 1.2 > Conclusion 1.1 > Methods 1.0 > Introduction 0.85
- Ingestion = high recall, Retrieval = medium precision, LLM = final filter
- Embedding-based dedup removes semantically duplicate chunks
- Source diversity cap (2 chunks/paper) prevents single-paper dominance
- Context budget: 4k chars, max 5 chunks, each truncated to 500 chars before budgeting
- Preprint page banners (bioRxiv/medRxiv) and margin line numbers are stripped before section detection
- Hallucination guard checks key-term overlap with source corpus
- Confidence scoring combines retrieval quality + citation density + context overlap
- Low temperature (0.2) for factual, deterministic output

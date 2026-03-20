# Alzheimer's RAG Research Assistant

A domain-specific, grounded research assistant for Alzheimer's disease literature. Built with section-aware chunking, FAISS retrieval, reranking, and local LLM inference via Ollama (phi3).

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

# 3. Pull the phi3 model
ollama pull phi3

# 4. Start Ollama (keep running in a separate terminal)
ollama serve
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
│  → embeddings     │  (all-MiniLM-L6-v2)
│  → FAISS index    │
└────────┬─────────┘
         │ saved to processed/
         ▼
┌──────────────────┐
│    Retrieval      │  query → enhance → FAISS top-k
│                   │  → rerank (score + keywords + section weight)
│                   │  → embedding dedup
│                   │  → diversify (max 2 per paper)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM (Ollama)     │  context trimming (4k char budget)
│  phi3 model       │  → structured prompt (cited bullet points)
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

# Step 4: Start the API server
uvicorn app.api:app --reload --port 8000
```

## API Endpoints

Once the server is running:

| Method | Endpoint   | Description                    |
|--------|-----------|--------------------------------|
| GET    | /health   | Check if retriever is loaded   |
| POST   | /query    | Ask a question (JSON body)     |
| GET    | /sources  | List all indexed papers        |
| GET    | /docs     | Interactive Swagger UI         |

Example query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the risk factors for Alzheimer?"}'
```

## Evaluation

`python -m app.evaluate` runs three checks:

- Retrieval — relevant chunks, source diversity, score thresholds
- Answers — grounding (context overlap), citation presence, answer length
- Safety — refuses medical advice, no fabricated claims

## Project Structure

```
├── data/               # Source PDFs (not tracked)
├── processed/          # chunks.json + embeddings.faiss (generated)
├── app/
│   ├── ingest.py       # PDF → chunks → embeddings → FAISS index
│   ├── retrieve.py     # Search → rerank → dedup → diversify
│   ├── llm.py          # Ollama/phi3 integration, prompt, guards
│   ├── evaluate.py     # Retrieval + answer + safety evaluation
│   ├── api.py          # FastAPI server
│   ├── main.py         # CLI entry point
│   └── utils.py        # Text cleaning, section detection, chunking
├── tests/
│   ├── audit_data.py   # Data audit utility
│   └── eval_results.md # Evaluation results log
├── .env                # HF_TOKEN (not tracked)
├── requirements.txt
└── .gitignore
```

## Design Decisions

- Section-aware chunking preserves academic structure (Abstract, Methods, Results, etc.)
- Section weighting: Results > Discussion > Abstract > Conclusion > Methods > Introduction
- Ingestion = high recall, Retrieval = medium precision, LLM = final filter
- Embedding-based dedup removes semantically duplicate chunks
- Source diversity cap (2 chunks/paper) prevents single-paper dominance
- Context budget: 4k chars, max 5 chunks, 500 char per chunk
- Hallucination guard checks key-term overlap with source corpus
- Confidence scoring combines retrieval quality + citation density + context overlap
- Low temperature (0.2) for factual, deterministic output

# Alzheimer's RAG Research Assistant

A domain-specific, grounded research assistant for Alzheimer's disease literature. Built with custom ingestion, section-aware chunking, FAISS retrieval, reranking, and structured prompting via Google Gemini.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key to .env
echo GOOGLE_API_KEY=your-key-here > .env
```

## How It Works

```
PDF files (data/)
    │
    ▼
┌──────────────┐
│   Ingestion   │  python -m app.ingest
│               │
│  PDF → text   │
│  → sections   │  (detects academic headings)
│  → chunks     │  (400-word, sentence-aware)
│  → embeddings │  (all-MiniLM-L6-v2)
│  → FAISS index│
└──────┬───────┘
       │ saved to processed/
       ▼
┌──────────────┐
│  Retrieval    │  query → enhance → FAISS top-k
│               │  → rerank (score + keywords + section weight)
│               │  → diversify (max 2 chunks per paper)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM (Gemini) │  context trimming (score-sorted, deduped)
│               │  → structured prompt (grounded, cited)
│               │  → safety constraints (no medical advice)
└──────────────┘
```

## Usage

```bash
# Step 1: Ingest PDFs (run once, or when you add new papers)
python -m app.ingest

# Step 2: Ask questions
python -m app.main "What biomarkers are used for early detection of Alzheimer's?"

# Step 3: Run evaluation suite
python -m app.evaluate
```

## Evaluation

`python -m app.evaluate` runs three checks:

- Retrieval — are top chunks relevant, diverse, from correct sections?
- Answers — are responses grounded, cited, well-structured?
- Safety — does the system refuse unsafe queries (medical advice, fabricated info)?

Prints a summary with pass/fail counts per layer.

## Project Structure

```
├── data/               # Source PDFs (not tracked in git)
├── processed/          # chunks.json + embeddings.faiss (generated)
├── app/
│   ├── ingest.py       # PDF → chunks → embeddings → FAISS index
│   ├── retrieve.py     # Search → rerank → diversify pipeline
│   ├── llm.py          # Gemini integration, prompt engineering
│   ├── evaluate.py     # Retrieval + answer + safety evaluation
│   ├── main.py         # CLI entry point
│   └── utils.py        # Text cleaning, section detection, chunking
├── .env                # API key (not tracked)
├── requirements.txt
└── .gitignore
```

## Key Design Decisions

- Section-aware chunking preserves academic structure (Abstract, Methods, Results, etc.)
- Reranking combines cosine similarity + keyword overlap + section weighting
- Source diversity cap (2 chunks/paper) prevents single-paper dominance
- Context trimming prioritizes highest-scored chunks within a 12k char budget
- Low temperature (0.2) for factual, deterministic output
- Prompt enforces strict grounding, citation anchoring, and medical safety

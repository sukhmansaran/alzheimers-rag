"""
Ingestion pipeline: PDF → sections → chunks → embeddings → FAISS index.
"""
import json
import os

import faiss
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.utils import (
    chunk_text_weighted, clean_text, detect_sections, get_pdf_paths,
    is_reference_noise, is_bad_chunk, is_answerable,
)

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", None)

PROCESSED_DIR = "processed"
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
INDEX_PATH = os.path.join(PROCESSED_DIR, "embeddings.faiss")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Sections to skip — only actual garbage
SKIP_SECTIONS = {"References", "Acknowledgments", "Preamble", "Keywords"}


def extract_text(pdf_path: str) -> str:
    """Extract clean text from a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n".join(pages)


def process_pdf(pdf_path: str) -> list[dict]:
    """Parse one PDF into metadata-rich chunks."""
    paper = os.path.basename(pdf_path)
    raw = extract_text(pdf_path)
    if not raw.strip():
        print(f"  ⚠ No text extracted: {paper}")
        return []

    # Detect sections on raw text (needs newlines for heading detection)
    sections = detect_sections(raw)
    all_chunks = []
    for sec in sections:
        # Skip only actual garbage sections
        if sec["section"] in SKIP_SECTIONS:
            continue
        # Chunk with section-aware weighting
        texts = chunk_text_weighted(sec["text"], sec["section"])
        for chunk_info in texts:
            cleaned = clean_text(chunk_info["text"])
            if not cleaned:
                continue
            if is_reference_noise(cleaned) or is_bad_chunk(cleaned):
                continue
            if not is_answerable(cleaned):
                continue
            if len(set(cleaned.split())) < 15:
                continue
            all_chunks.append({
                "text": cleaned,
                "paper": paper,
                "section": chunk_info["section"],
                "weight": chunk_info["weight"],
                "word_count": len(cleaned.split()),
            })
    print(f"  {paper} → chunks kept: {len(all_chunks)}")
    return all_chunks


def run_ingestion():
    """Full ingestion pipeline."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Parse all PDFs
    pdfs = get_pdf_paths()
    print(f"Found {len(pdfs)} PDFs")

    all_chunks = []
    for pdf_path in pdfs:
        print(f"  Processing: {os.path.basename(pdf_path)}")
        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)

    # Assign IDs
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"chunk_{i:04d}"

    print(f"Total chunks: {len(all_chunks)}")

    # 2. Save chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks → {CHUNKS_PATH}")

    # 3. Embed
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    # 4. FAISS index (cosine similarity via normalized IP)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index → {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()

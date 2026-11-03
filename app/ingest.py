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
    PROJECT_ROOT,
    chunk_text_weighted,
    clean_text,
    detect_sections,
    get_pdf_paths,
    is_answerable,
    is_bad_chunk,
    is_reference_noise,
    strip_layout_noise,
)

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", None)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
INDEX_PATH = os.path.join(PROCESSED_DIR, "embeddings.faiss")
INDEX_META_PATH = os.path.join(PROCESSED_DIR, "index_meta.json")

# bge-small reads 512 tokens per chunk (MiniLM truncated at 256, ~1/3 of a chunk) and scored
# best on the labelled eval. Queries get bge's retrieval instruction; passages are embedded as-is.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Sections to skip — only actual garbage
SKIP_SECTIONS = {"References", "Acknowledgments", "Preamble", "Keywords"}


# Two-column detection: in single-column text most lines have a word crossing the page's
# vertical midline (measured 0.2-0.8 straddling words per line); two-column pages have a gutter
# there (measured 0.03-0.05). Table/figure pages can also look two-column, so column-aware
# extraction is only used when most pages of the document are two-column.
MIN_WORDS_FOR_LAYOUT = 80
# pdfplumber's default 3pt gap merged words in tightly set PDFs ("theywerenon-originalreviews");
# 2pt separates them and left word counts of the other papers unchanged (measured).
X_TOLERANCE = 2
TWO_COLUMN_MAX_STRADDLE = 0.15
TWO_COLUMN_DOC_RATIO = 0.5


def _line_groups(words: list[dict]) -> list[list[dict]]:
    """Group words into lines by vertical position (3pt tolerance), top to bottom."""
    lines: dict[int, list[dict]] = {}
    for w in words:
        lines.setdefault(round(w["top"] / 3), []).append(w)
    return [sorted(ln, key=lambda w: w["x0"]) for _, ln in sorted(lines.items())]


def _is_two_column(page) -> bool:
    words = page.extract_words(x_tolerance=X_TOLERANCE)
    if len(words) < MIN_WORDS_FOR_LAYOUT:
        return False
    mid = page.width / 2
    straddling = sum(1 for w in words if w["x0"] < mid < w["x1"])
    left = sum(1 for w in words if w["x1"] <= mid)
    right = sum(1 for w in words if w["x0"] >= mid)
    n_lines = len(_line_groups(words))
    return straddling / max(n_lines, 1) < TWO_COLUMN_MAX_STRADDLE and min(left, right) > 0.25 * (left + right)


def _extract_two_column(page) -> str:
    """Full-width lines at the top (banner, title) first, then the left column, then the right."""
    mid = page.width / 2
    header, left, right = [], [], []
    in_header = True
    for line in _line_groups(page.extract_words(x_tolerance=X_TOLERANCE)):
        if any(w["x0"] < mid < w["x1"] for w in line):
            (header if in_header else left).append(" ".join(w["text"] for w in line))
            continue
        in_header = False
        lw = [w["text"] for w in line if w["x1"] <= mid]
        rw = [w["text"] for w in line if w["x0"] >= mid]
        if lw:
            left.append(" ".join(lw))
        if rw:
            right.append(" ".join(rw))
    return "\n".join(header + left + right)


def extract_text(pdf_path: str) -> str:
    """Extract clean text from a PDF, reading two-column documents column by column."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        layout = [
            (p, _is_two_column(p), len(p.extract_words(x_tolerance=X_TOLERANCE)) >= MIN_WORDS_FOR_LAYOUT)
            for p in pdf.pages
        ]
        text_pages = [two for _, two, has_text in layout if has_text]
        two_column_doc = bool(text_pages) and sum(text_pages) / len(text_pages) > TWO_COLUMN_DOC_RATIO
        for page, two, _ in layout:
            t = _extract_two_column(page) if (two_column_doc and two) else page.extract_text(x_tolerance=X_TOLERANCE)
            if t:
                pages.append(t)
    return "\n".join(pages)


def process_pdf(pdf_path: str) -> list[dict]:
    """Parse one PDF into metadata-rich chunks."""
    paper = os.path.basename(pdf_path)
    raw = strip_layout_noise(extract_text(pdf_path))
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
        try:
            chunks = process_pdf(pdf_path)
        except Exception as e:  # one corrupt PDF shouldn't abort the whole run
            print(f"  ⚠ Failed to process {os.path.basename(pdf_path)}: {e}")
            continue
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No chunks produced — add readable PDFs to data/ and re-run.")

    # Assign IDs
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"chunk_{i:04d}"

    print(f"Total chunks: {len(all_chunks)}")

    # 2. Embed
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    # 3. FAISS index (cosine similarity via normalized IP)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index → {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")

    # 4. Save chunks last: their order must match the index rows
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks → {CHUNKS_PATH}")

    # 5. Record which model built the index (checked at load time)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"embed_model": EMBED_MODEL, "chunks": len(all_chunks), "dim": dim}, f, indent=2)

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()

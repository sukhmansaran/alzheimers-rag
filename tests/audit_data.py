"""
Data quality audit for the RAG pipeline.
Inspects chunks, measures quality, checks retrieval hits, and section distribution.

Usage:
    python -m tests.audit_data
"""
import json
import random
from collections import Counter

from app.retrieve import load_retriever, search
from app.ingest import CHUNKS_PATH


# ─────────────────────────────────────────────────────────────
# Step 1: Inspect random chunks
# ─────────────────────────────────────────────────────────────

def inspect_chunks(chunks, n=10):
    print("\n" + "=" * 60)
    print(f"INSPECTING {n} RANDOM CHUNKS")
    print("=" * 60)

    for c in random.sample(chunks, min(n, len(chunks))):
        print("\n---")
        print("PAPER:", c["paper"])
        print("SECTION:", c["section"])
        print("WORDS:", c["word_count"])
        print("TEXT:\n", c["text"][:500])


# ─────────────────────────────────────────────────────────────
# Step 2: Chunk quality metrics
# ─────────────────────────────────────────────────────────────

def analyze_chunks(chunks):
    total = len(chunks)
    short = sum(1 for c in chunks if len(c["text"]) < 100)
    noisy = sum(1 for c in chunks if "et al" in c["text"] or "doi" in c["text"])
    numeric = sum(
        1 for c in chunks
        if sum(x.isdigit() for x in c["text"]) > 0.3 * len(c["text"])
    )

    print("\n" + "=" * 60)
    print("CHUNK QUALITY REPORT")
    print("=" * 60)
    print(f"Total chunks:      {total}")
    print(f"Too short (<100):  {short} ({short/total:.1%})")
    print(f"Noisy (citations): {noisy} ({noisy/total:.1%})")
    print(f"Numeric garbage:   {numeric} ({numeric/total:.1%})")

    clean = total - short - noisy - numeric
    print(f"Clean:             {clean} ({clean/total:.1%})")


# ─────────────────────────────────────────────────────────────
# Step 3: Inspect top retrieved chunks for sample queries
# ─────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "What biomarkers are used for early detection of Alzheimer's?",
    "What increases risk of Alzheimer's?",
]


def inspect_hits(model, index, chunks):
    print("\n" + "=" * 60)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 60)

    for q in SAMPLE_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"QUERY: {q}")
        print("─" * 60)

        hits = search(q, model, index, chunks, k=5)
        for r in hits:
            print(f"\n  [{r['rank']}] Score: {r['score']:.3f}")
            print(f"  {r['paper']} | {r['section']}")
            print(f"  {r['text'][:400]}")


# ─────────────────────────────────────────────────────────────
# Step 4: Section distribution
# ─────────────────────────────────────────────────────────────

def section_stats(chunks):
    counts = Counter(c["section"] for c in chunks)

    print("\n" + "=" * 60)
    print("SECTION DISTRIBUTION")
    print("=" * 60)
    for k, v in counts.most_common():
        bar = "█" * (v // 2)
        print(f"  {k:30s} {v:4d}  {bar}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    # Load chunks from file
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    inspect_chunks(chunks, n=10)
    analyze_chunks(chunks)
    section_stats(chunks)

    # Load retriever for hit inspection
    model, index, all_chunks = load_retriever()
    inspect_hits(model, index, all_chunks)


if __name__ == "__main__":
    main()

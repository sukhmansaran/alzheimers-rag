"""
Entry point: query the RAG pipeline.

Usage:
    python -m app.main "What are the risk factors for Alzheimer's?"
"""
import sys

from app.retrieve import load_retriever, search
from app.llm import ask


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.main \"your question\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n🔍 Query: {query}\n")

    # Load retriever
    model, index, chunks = load_retriever()

    # Retrieve
    results = search(query, model, index, chunks, k=5)

    print("📚 Sources retrieved:")
    for r in results:
        print(f"  [{r['rank']}] {r['paper']} — {r['section']} (dist: {r['score']:.4f})")

    # Generate answer
    print("\n🤖 Generating answer...\n")
    answer = ask(query, results)
    print(answer)


if __name__ == "__main__":
    main()

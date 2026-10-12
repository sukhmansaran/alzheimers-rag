"""
Entry point: query the RAG pipeline.

Usage:
    python -m app.main "What are the risk factors for Alzheimer's?"
"""
import sys

from app.llm import DISCLAIMER, LLMError, ask
from app.retrieve import load_retriever, search


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # β, ≥, emoji in papers/answers; safe when redirected
    if len(sys.argv) < 2:
        print("Usage: python -m app.main \"your question\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n🔍 Query: {query}\n")

    # Load retriever
    model, index, chunks = load_retriever()

    # Retrieve
    results = search(query, model, index, chunks, k=5)
    if not results:
        print("No relevant sources found for this question.")
        sys.exit(0)

    print("📚 Sources retrieved:")
    for r in results:
        print(f"  [{r['rank']}] {r['paper']} — {r['section']} (similarity: {r['score']:.4f})")

    # Generate answer
    print("\n🤖 Generating answer...\n")
    try:
        result = ask(query, results)
    except LLMError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(result["answer"])
    print(f"\nConfidence: {result['confidence']:.2f}")
    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")
    print(f"\n⚠️  {DISCLAIMER}")


if __name__ == "__main__":
    main()

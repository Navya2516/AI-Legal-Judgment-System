import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from query_expansion_retrieval import load_resources, retrieve
from backend.app.services.llm_service import generate_grounded_answer


def main():
    question = "What constitutional amendments are involved in this case?"

    print("=" * 70)
    print("LEGAL RAG ANSWER TEST")
    print("=" * 70)

    print("\nLoading FAISS resources...")
    index, metadata, model = load_resources()

    print("Retrieving relevant evidence...")
    expanded_queries, retrieved_chunks = retrieve(
    question,
    index,
    metadata,
    model
    )

    print(f"Retrieved {len(retrieved_chunks)} evidence chunks.")
    print(f"Expanded queries: {len(expanded_queries)}")
    print(f"Retrieved {len(retrieved_chunks)} evidence chunks.")

    print("\nGenerating grounded answer using Gemini...")
    answer = generate_grounded_answer(
        question,
        retrieved_chunks
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("GEMINI ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RAG TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
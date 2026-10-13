import json
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent)
)

from query_expansion_retrieval import (
    load_resources,
    retrieve
)



OUTPUT_PATH = Path(
    "data/processed/rag_context_test.json"
)


def build_context(
    question,
    results
):

    sources = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        sources.append({
            "source_rank": rank,
            "chunk_id": result["chunk_id"],
            "page_number": result["page_number"],
            "retrieval_score": round(
                result["final_score"],
                4
            ),
            "text": result["text"]
        })

    return {
        "question": question,
        "sources": sources
    }


def main():

    print("Loading retrieval resources...")

    index, metadata, model = load_resources()

    question = (
        "What constitutional amendments "
        "are involved in this case?"
    )

    expanded_queries, results = retrieve(
        question,
        index,
        metadata,
        model
    )

    context = build_context(
        question,
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            context,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("RAG CONTEXT CREATED")
    print("=" * 60)

    print(
        f"Original question : {question}"
    )

    print()
    print("Expanded queries:")

    for query in expanded_queries:
        print(f"- {query}")

    print()
    print("Retrieved evidence:")

    for source in context["sources"]:

        print()
        print(
            f"Source {source['source_rank']}"
        )

        print(
            f"Page  : {source['page_number']}"
        )

        print(
            f"Chunk : {source['chunk_id']}"
        )

        print(
            f"Score : {source['retrieval_score']}"
        )

        print(
            f"Text  : "
            f"{source['text'][:300]}..."
        )

    print()
    print(
        f"Context saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
from backend.app.services.rag_service import (
    load_resources,
    retrieve
)


DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"


def test_question(question):
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    print("\nLoading resources...")

    index, metadata, model = load_resources(
        DOCUMENT_ID
    )

    expanded_queries, chunks = retrieve(
        question,
        index,
        metadata,
        model
    )

    print("\n" + "=" * 80)
    print("EXPANDED QUERIES:")
    print("=" * 80)

    for query in expanded_queries:
        print(f"- {query}")

    print("\n" + "=" * 80)
    print("RETRIEVED EVIDENCE:")
    print("=" * 80)

    for i, chunk in enumerate(
        chunks,
        start=1
    ):
        print("\n" + "-" * 80)

        print(
            f"SOURCE {i}"
        )

        print(
            f"PAGE: {chunk.get('page_number')}"
        )

        print(
            f"SCORE: {chunk.get('final_score')}"
        )

        print(
            f"CHUNK ID: {chunk.get('chunk_id')}"
        )

        print("\nTEXT:\n")

        print(
            chunk.get(
                "text",
                ""
            )[:2500]
        )


# ============================================================
# TEST 1
# ============================================================

test_question(
    "What did the 24th Amendment change?"
)


# ============================================================
# TEST 2
# ============================================================

test_question(
    "What was Parliament's power to amend the Constitution under Article 368?"
)
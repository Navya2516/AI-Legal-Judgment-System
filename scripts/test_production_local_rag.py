from backend.app.services.rag_service import (
    answer_question
)


DOCUMENT_ID = (
    "0adde2b14334482d83ae091f61003bc6"
)


QUESTION = (
    "What constitutional amendments were considered "
    "in the Kesavananda Bharati case?"
)


print("\n")
print("=" * 70)
print("PRODUCTION RAG + LOCAL LLM TEST")
print("=" * 70)


result = answer_question(
    document_id=DOCUMENT_ID,
    question=QUESTION
)


print("\nQuestion:")
print(
    result["question"]
)


print("\nExpanded Queries:")

for query in result[
    "expanded_queries"
]:

    print(
        f"- {query}"
    )


print("\nRetrieved Evidence:")

for i, evidence in enumerate(
    result["retrieved_evidence"],
    1
):

    print(
        f"\n--- Evidence {i} ---"
    )

    print(
        f"Page: "
        f"{evidence['page_number']}"
    )

    print(
        f"Chunk: "
        f"{evidence['chunk_id']}"
    )

    print(
        f"Score: "
        f"{evidence['score']}"
    )


print("\n")
print("=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(
    result["answer"]
)

print("\n")
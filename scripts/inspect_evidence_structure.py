from backend.app.services.rag_service import (
    load_resources,
    retrieve
)

from backend.app.services.evidence_service import (
    build_evidence_facts
)


DOCUMENT_ID = (
    "0adde2b14334482d83ae091f61003bc6"
)

QUESTION = (
    "What constitutional amendments were considered "
    "in the Kesavananda Bharati case?"
)


index, metadata, model = load_resources(
    DOCUMENT_ID
)

queries, retrieved_chunks = retrieve(
    QUESTION,
    index,
    metadata,
    model
)

facts = build_evidence_facts(
    retrieved_chunks
)

print("\n")
print("=" * 70)
print("EVIDENCE FACT STRUCTURE")
print("=" * 70)

for key, value in facts.items():

    print(
        f"\nKEY: {key}"
    )

    print(
        f"TYPE: {type(value).__name__}"
    )

    print(
        f"VALUE: {value}"
    )
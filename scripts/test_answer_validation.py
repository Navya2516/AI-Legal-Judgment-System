from backend.app.services.rag_service import (
    load_resources,
    retrieve
)

from backend.app.services.evidence_service import (
    build_evidence_facts
)

from backend.app.services.answer_validation_service import (
    validate_answer
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

_, retrieved_chunks = retrieve(
    QUESTION,
    index,
    metadata,
    model
)

facts = build_evidence_facts(
    retrieved_chunks
)


answer = """
The case considered the 24th Amendment [Page 178],
25th Amendment [Page 282],
26th Amendment [Page 111],
and 29th Amendment [Page 111].
"""


result = validate_answer(
    QUESTION,
    answer,
    facts
)


print("\n")
print("=" * 60)
print("ANSWER VALIDATION")
print("=" * 60)

print(
    f"\nCoverage: "
    f"{result['coverage_score']}%"
)

print(
    f"Complete: "
    f"{result['is_complete']}"
)

print(
    f"\nDetected amendments:"
)

for item in result[
    "detected_amendments"
]:
    print(
        f"  ✓ {item}"
    )

print(
    f"\nMissing amendments:"
)

for item in result[
    "missing_amendments"
]:
    print(
        f"  ✗ {item}"
    )

print(
    f"\nPage citations: "
    f"{result['page_citations']}"
)
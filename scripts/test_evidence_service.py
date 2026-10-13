from backend.app.services.rag_service import (
    load_resources,
    retrieve
)

from backend.app.services.evidence_service import (
    build_evidence_facts,
    validate_answer_against_evidence
)


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

QUESTION = (
    "What constitutional amendments were considered "
    "in the Kesavananda Bharati case?"
)


# ============================================================
# LOAD RESOURCES
# ============================================================

print("\nLoading resources...\n")

index, metadata, model = load_resources(
    DOCUMENT_ID
)


# ============================================================
# RETRIEVE
# ============================================================

print("\nRunning retrieval...\n")

expanded_queries, retrieved_chunks = retrieve(
    QUESTION,
    index,
    metadata,
    model
)


# ============================================================
# BUILD FACTS
# ============================================================

facts = build_evidence_facts(
    retrieved_chunks
)


# ============================================================
# DISPLAY STRUCTURED FACTS
# ============================================================

print("\n")
print("=" * 70)
print("STRUCTURED EVIDENCE FACTS")
print("=" * 70)


print("\nAmendments:")

for amendment in facts["amendments"]:

    print(
        f"\n{amendment['name']}"
    )

    print(
        f"Pages: {amendment['evidence_pages']}"
    )

    print(
        f"Chunks: {amendment['evidence_chunks']}"
    )


print("\nArticles:")

for article in facts["articles"]:

    print(
        f"{article['name']} "
        f"→ pages {article['evidence_pages']}"
    )


print("\nSections:")

for section in facts["sections"]:

    print(
        f"{section['name']} "
        f"→ pages {section['evidence_pages']}"
    )


print("\nAll evidence pages:")

print(
    facts["pages"]
)


# ============================================================
# VALIDATION TEST
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION TEST")
print("=" * 70)


test_answer = """
The case considered the 24th Amendment, 25th Amendment,
26th Amendment and 29th Amendment.
Article 368 was also central to the question of Parliament's
constitutional amending power.
"""


validation = validate_answer_against_evidence(
    test_answer,
    retrieved_chunks
)


print("\nValidation result:")

print(validation)
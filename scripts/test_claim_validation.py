from backend.app.services.rag_service import (
    load_resources,
    retrieve
)

from backend.app.services.claim_validation_service import (
    validate_claims
)


DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

QUESTION = (
    "What constitutional amendments were considered "
    "in the Kesavananda Bharati case?"
)


# ------------------------------------------------------------
# Qwen answer from the previous test
# ------------------------------------------------------------

ANSWER = """
The constitutional amendments considered in the Kesavananda
Bharati case include:

1. The 24th Amendment: This amendment was explicitly mentioned
on pages [178, 282, 434, 427]. It explicitly changed Article 13
by inserting Sub-article (4) that nullified the operation of
Article 13 by amending Article 13 by a new Sub-article (4).
This amendment was a clarifying amendment of Article 368 to
Article 13 and did not affect any essential feature of the
Constitution.

2. The 25th Amendment: This amendment was explicitly mentioned
on pages [282, 427]. It introduced some abridgement of the
fundamental right to property.

3. The 29th Amendment: This amendment was explicitly mentioned
on pages [282, 111]. It was considered valid but the question
of its validity was deferred.

4. The 26th Amendment: This amendment was explicitly mentioned
on pages [111]. The question was deferred to be decided later.

The Kesavananda Bharati case is significant as it expanded the
power of Parliament to amend the Constitution without the
limitations of the fundamental rights.
"""


# ============================================================
# LOAD RAG
# ============================================================

print("\nLoading resources...\n")

index, metadata, model = load_resources(
    DOCUMENT_ID
)


# ============================================================
# RETRIEVE
# ============================================================

print("\nRetrieving evidence...\n")

_, retrieved_chunks = retrieve(
    QUESTION,
    index,
    metadata,
    model
)


# ============================================================
# VALIDATE
# ============================================================

result = validate_claims(
    ANSWER,
    retrieved_chunks
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 70)
print("CLAIM-LEVEL VALIDATION")
print("=" * 70)

print(
    f"\nTotal claims: "
    f"{result['total_claims']}"
)

print(
    f"Supported claims: "
    f"{result['supported_claims']}"
)

print(
    f"Unsupported claims: "
    f"{result['unsupported_claims']}"
)

print(
    f"Faithfulness score: "
    f"{result['faithfulness_score']:.2%}"
)


print("\nIndividual claims:\n")

for i, item in enumerate(
    result["claims"],
    1
):

    print(
        f"{i}. {item['claim']}"
    )

    print(
        f"   Supported: "
        f"{item['supported']}"
    )

    print(
        f"   Overlap: "
        f"{item['overlap_score']}"
    )

    print(
        f"   Evidence page: "
        f"{item['evidence_page']}"
    )

    print()
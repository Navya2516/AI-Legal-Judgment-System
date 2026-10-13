import ollama

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

MODEL_NAME = "qwen2.5:1.5b"

QUESTION = (
    "What constitutional amendments were considered "
    "in the Kesavananda Bharati case?"
)


# ============================================================
# LOAD RESOURCES
# ============================================================

print("\nLoading judgment resources...\n")

index, metadata, embedding_model = load_resources(
    DOCUMENT_ID
)


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

print("\nRunning RAG retrieval...\n")

expanded_queries, retrieved_chunks = retrieve(
    QUESTION,
    index,
    metadata,
    embedding_model
)


# ============================================================
# BUILD STRUCTURED EVIDENCE FACTS
# ============================================================

evidence_facts = build_evidence_facts(
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

for amendment in evidence_facts["amendments"]:

    print(
        f"- {amendment['name']}"
    )

    print(
        f"  Pages: {amendment['evidence_pages']}"
    )


print("\nArticles:")

for article in evidence_facts["articles"]:

    print(
        f"- {article['name']}"
        f" → pages {article['evidence_pages']}"
    )


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

evidence_context = ""

for i, chunk in enumerate(
    retrieved_chunks,
    1
):

    evidence_context += f"""

[EVIDENCE {i}]
Page: {chunk['page_number']}
Chunk ID: {chunk['chunk_id']}

{chunk['text']}
"""


# ============================================================
# BUILD STRUCTURED FACT CONTEXT
# ============================================================

amendment_facts = "\n".join(
    f"- {item['name']}: "
    f"explicitly mentioned on pages "
    f"{item['evidence_pages']}"
    for item in evidence_facts["amendments"]
)


article_facts = "\n".join(
    f"- {item['name']}: "
    f"explicitly mentioned on pages "
    f"{item['evidence_pages']}"
    for item in evidence_facts["articles"]
)


structured_facts = f"""
AMENDMENTS EXPLICITLY DETECTED:

{amendment_facts}

ARTICLES EXPLICITLY DETECTED:

{article_facts}
"""


# ============================================================
# STRICT EVIDENCE-GROUNDED PROMPT
# ============================================================

prompt = f"""
You are an evidence-grounded legal judgment explanation
assistant.

Your task is to answer the user's question using ONLY
the retrieved evidence and structured evidence facts below.

DO NOT use your pretrained knowledge.

DO NOT add information that is not present in the evidence.

IMPORTANT RULES:

1. Every factual claim must be supported by the retrieved
   evidence.

2. The structured evidence facts identify legal terms that
   were explicitly detected in the retrieved text.

3. If an amendment appears in the structured facts, do not
   omit it when the question asks which amendments were
   considered or mentioned.

4. Do not invent dates.

5. Do not invent the effects of an amendment.

6. Do not invent the legal status of an amendment.

7. Do not infer information that is not explicitly supported.

8. If the evidence does not establish something, say:
   "The retrieved evidence does not establish this."

9. Keep the answer concise and easy to understand.

10. Do not provide legal advice.

11. Do not predict the result of a future case.

USER QUESTION:

{QUESTION}

STRUCTURED EVIDENCE FACTS:

{structured_facts}

RETRIEVED JUDGMENT EVIDENCE:

{evidence_context}

ANSWER REQUIREMENTS:

- Directly answer the question.
- List every amendment identified in the structured evidence
  that is relevant to the question.
- Explain each amendment ONLY to the extent supported by
  the retrieved evidence.
- If the retrieved evidence says that an amendment's validity
  was deferred, distinguish that from an amendment whose
  validity was decided.
- Do not supply historical dates unless they appear in the
  evidence.
"""


# ============================================================
# CALL QWEN
# ============================================================

print("\nGenerating answer with Qwen...\n")

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


answer = response["message"]["content"]


# ============================================================
# VALIDATE ANSWER
# ============================================================

validation = validate_answer_against_evidence(
    answer,
    retrieved_chunks
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("GROUNDED RAG + STRUCTURED EVIDENCE + LOCAL QWEN")
print("=" * 70)

print("\nQuestion:")
print(QUESTION)

print("\nAnswer:")
print(answer)

print("\n")
print("=" * 70)
print("ANSWER VALIDATION")
print("=" * 70)

print(
    f"\nSupported: "
    f"{validation['is_supported']}"
)

print(
    f"Unsupported count: "
    f"{validation['unsupported_count']}"
)

if validation["unsupported_articles"]:

    print(
        "\nUnsupported Articles:"
    )

    for item in validation[
        "unsupported_articles"
    ]:

        print(
            f"- {item}"
        )


if validation["unsupported_sections"]:

    print(
        "\nUnsupported Sections:"
    )

    for item in validation[
        "unsupported_sections"
    ]:

        print(
            f"- {item}"
        )


if validation["unsupported_amendments"]:

    print(
        "\nUnsupported Amendments:"
    )

    for item in validation[
        "unsupported_amendments"
    ]:

        print(
            f"- {item}"
        )
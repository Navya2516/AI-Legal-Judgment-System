import os
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "local"
).lower()

LOCAL_LLM_MODEL = os.getenv(
    "LOCAL_LLM_MODEL",
    "qwen2.5:1.5b"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if LLM_PROVIDER == "gemini":

    from google import genai

    API_KEY = os.getenv(
        "GEMINI_API_KEY"
    )

    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found in .env"
        )

    gemini_client = genai.Client(
        api_key=API_KEY
    )


# ============================================================
# EVIDENCE SERVICE
# ============================================================

from backend.app.services.evidence_service import (
    build_evidence_facts
)


# ============================================================
# BUILD RAW EVIDENCE CONTEXT
# ============================================================

def build_evidence_context(
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Convert retrieved chunks into a clear evidence context.

    Each source contains:
    - page number
    - chunk ID
    - retrieval score
    - judgment text
    """

    evidence_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        page = chunk.get(
            "page_number",
            "Unknown"
        )

        chunk_id = chunk.get(
            "chunk_id",
            "Unknown"
        )

        score = chunk.get(
            "final_score",
            chunk.get(
                "score",
                "Unknown"
            )
        )

        text = chunk.get(
            "text",
            ""
        )

        evidence_parts.append(
            f"""
============================================================
SOURCE {i}
============================================================
PAGE: {page}
CHUNK ID: {chunk_id}
RETRIEVAL SCORE: {score}

{text}
"""
        )

    return "\n".join(
        evidence_parts
    )


# ============================================================
# BUILD STRUCTURED LEGAL FACTS
# ============================================================

def build_structured_evidence(
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Convert deterministic evidence extraction results
    into a compact text representation for the LLM.
    """

    facts = build_evidence_facts(
        retrieved_chunks
    )

    sections = []

    # ========================================================
    # AMENDMENTS
    # ========================================================

    amendments = facts.get(
        "amendments",
        []
    )

    if amendments:

        text = (
            "\nAMENDMENTS FOUND IN RETRIEVED EVIDENCE:\n"
        )

        for item in amendments:

            name = item.get(
                "name",
                "Unknown Amendment"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            chunks = item.get(
                "evidence_chunks",
                []
            )

            text += (
                f"- {name}\n"
                f"  Evidence Pages: {pages}\n"
                f"  Evidence Chunks: {chunks}\n"
            )

        sections.append(
            text
        )

    # ========================================================
    # ARTICLES
    # ========================================================

    articles = facts.get(
        "articles",
        []
    )

    if articles:

        text = (
            "\nARTICLES FOUND IN RETRIEVED EVIDENCE:\n"
        )

        for item in articles:

            name = item.get(
                "name",
                "Unknown Article"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            chunks = item.get(
                "evidence_chunks",
                []
            )

            text += (
                f"- {name}\n"
                f"  Evidence Pages: {pages}\n"
                f"  Evidence Chunks: {chunks}\n"
            )

        sections.append(
            text
        )

    # ========================================================
    # SECTIONS
    # ========================================================

    legal_sections = facts.get(
        "sections",
        []
    )

    if legal_sections:

        text = (
            "\nSECTIONS FOUND IN RETRIEVED EVIDENCE:\n"
        )

        for item in legal_sections:

            name = item.get(
                "name",
                "Unknown Section"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            chunks = item.get(
                "evidence_chunks",
                []
            )

            text += (
                f"- {name}\n"
                f"  Evidence Pages: {pages}\n"
                f"  Evidence Chunks: {chunks}\n"
            )

        sections.append(
            text
        )

    # ========================================================
    # EVIDENCE SUMMARY
    # ========================================================

    evidence_count = facts.get(
        "evidence_count",
        len(retrieved_chunks)
    )

    pages = facts.get(
        "pages",
        []
    )

    sections.append(
        "\nEVIDENCE SUMMARY:\n"
        f"- Retrieved evidence chunks: {evidence_count}\n"
        f"- Evidence pages: {pages}\n"
    )

    return "\n".join(
        sections
    )


# ============================================================
# GROUNDED PROMPT
# ============================================================


def build_grounded_prompt(
    question: str,
    evidence: str,
    structured_facts: str
) -> str:
    """
    Build a strict evidence-grounded prompt for legal QA.

    The model must answer only from retrieved judgment evidence
    and must distinguish arguments from judicial conclusions.
    """

    return f"""
You are an evidence-grounded legal judgment explanation assistant.

Your task is to answer the user's question using ONLY the
retrieved excerpts from the judgment provided below.

========================
USER QUESTION
========================

{question}

========================
STRUCTURED EVIDENCE
========================

{structured_facts}

========================
RETRIEVED JUDGMENT EVIDENCE
========================

{evidence}

========================
STRICT RULES
========================

1. ANSWER THE EXACT QUESTION
--------------------------------
Answer what the user actually asked.

Do not change an explanatory question into a list question.

Example:

Question:
"What did the 24th Amendment change?"

Answer:
Explain the changes made by the 24th Amendment.

Do NOT simply list every amendment mentioned in the judgment.


2. USE ONLY THE RETRIEVED EVIDENCE
--------------------------------
Do not use outside legal knowledge.

Do not add facts, provisions, dates, cases, effects,
interpretations, or conclusions that are not supported
by the retrieved evidence.





3. DISTINGUISH ARGUMENTS FROM COURT FINDINGS
--------------------------------
This is extremely important.

The retrieved judgment excerpts may contain:

- arguments made by petitioners
- arguments made by respondents
- submissions by counsel
- judicial observations
- judicial reasoning
- conclusions/findings

Treat these as DIFFERENT categories.

A statement made by a party or counsel MUST NOT be presented
as something the Court held.

Pay special attention to attribution phrases such as:

- "it was urged..."
- "it was strenuously urged..."
- "it was submitted..."
- "it was argued..."
- "counsel contended..."
- "on behalf of the Union..."
- "on behalf of the States..."
- "the petitioner submitted..."
- "the respondents contended..."

These phrases normally introduce an argument or submission,
NOT a Court finding.

If an excerpt contains an argument followed by a Court response,
preserve both parts.

For example:

"It was strenuously urged on behalf of the Union and the States
that..."

followed by:

"We are unable to accept this contention."

means that the first statement is the parties' argument,
while the second statement is the Court's response.

Do NOT collapse both into a general statement such as:

"The Court held that the parties' argument was correct."

Similarly:

"The petitioner argued that X"

must NOT become:

"The Court held that X."


4. DO NOT INFER A COURT HOLDING
--------------------------------
Only describe something as a Court finding, holding,
conclusion, or decision when the retrieved evidence
explicitly supports that characterization.

If the evidence contains only an argument or submission,
report it as an argument or submission.

If the evidence contains both an argument and a response,
describe the relationship between them.

If the Court's final position is not clear from the retrieved
excerpts, DO NOT infer it from general legal knowledge.

If necessary, say:

"The retrieved excerpts show that this was argued before the
Court, but they do not provide enough evidence here to state
the Court's final position reliably."

Never convert an argument into a holding merely because the
argument appears close to a judicial statement.


    5. ARTICLE 368 QUESTIONS
    --------------------------------
    When the question concerns Article 368, first identify what
    the retrieved evidence clearly establishes.

    The answer should distinguish between:

    A. THE GENERAL AMENDING POWER
       - Article 368 as the source of the power to amend
         the Constitution.
       - The special form and manner prescribed by Article 368.

    B. THE NATURE OF THE POWER
       - Parliament and State Legislatures exercising
         constituent power when functioning under Article 368.
       - The distinction between constituent power and ordinary
         legislative power under Articles 245-248.

    C. THE EXTENT OF THE POWER
       - Whether the power extends to particular constitutional
         provisions, including Fundamental Rights.
       - Any limitations on the amending power.

    Do NOT assume that evidence discussing category C establishes
    a final answer to category C unless the retrieved excerpt
    explicitly provides the Court's conclusion.

    If the evidence clearly establishes the general amending power
    but does not provide enough evidence to establish the complete
    extent of that power, answer the general Article 368 question
    and explicitly state that the retrieved excerpts do not
    establish the complete extent.

    IMPORTANT:

    Do NOT say that Article 368 only provides a procedure.

    Do NOT say that Parliament has no power to amend Fundamental
    Rights unless the retrieved evidence explicitly supports that
    conclusion.

    Do NOT say that Parliament has unrestricted power to amend
    Fundamental Rights unless the retrieved evidence explicitly
    supports that conclusion.

    Do NOT treat a question raised in the judgment as the Court's
    final answer to that question.

    Do NOT infer a constitutional limitation merely because the
    retrieved evidence discusses arguments about such a limitation.

    If an excerpt contains an argument followed by a Court response,
    preserve that distinction.


6. AMENDMENT QUESTIONS
--------------------------------
When the question asks about a specific amendment:

Focus primarily on that amendment.

Do not describe the effects of another amendment unless
it is directly necessary to explain the requested amendment.


7. LEGAL EFFECT
--------------------------------
Do not assume that mentioning a provision automatically
establishes its legal effect.

For example:

"Article 13 was mentioned"

does not automatically mean:

"The amendment was invalid under Article 13."

The retrieved evidence must support the legal effect.


8. PRESERVE THE JUDGMENT'S MEANING
--------------------------------
Do not combine statements from different parts of the judgment
in a way that creates a new legal conclusion.

If two retrieved excerpts represent different opinions,
arguments, or stages of reasoning, keep them distinct.


9. WHEN EVIDENCE IS INSUFFICIENT
--------------------------------
If the retrieved evidence does not clearly answer the question,
say:

"The retrieved judgment evidence does not provide enough
information to answer this question reliably."

Do not guess.


10. ANSWER LENGTH
--------------------------------
Give a concise answer of approximately 3-5 sentences.

For a question asking for identified items, a short bullet list
is allowed.


11. CITATIONS
--------------------------------
Do not create page citations yourself.

The application will add page citations separately.


12. NO PERSONALIZED LEGAL ADVICE
--------------------------------
Do not provide legal advice, predict case outcomes,
recommend a litigation strategy, or claim that a particular
defense will succeed.

This system explains the judgment; it does not replace a lawyer.

    ========================
    FINAL SELF-CHECK
    ========================

    Before producing the answer, silently check:

    - Did I answer the exact question?
    - Did I use only retrieved evidence?
    - Did I accidentally turn an argument into a Court finding?
    - Did I merge different opinions?
    - Did I introduce outside legal knowledge?
    - Did I claim a legal effect that the evidence does not establish?
    - Did I focus on the requested Article/amendment?
    - Did I distinguish the general amendment power from the
      specific question of its extent?
    - Did I avoid answering a broader Fundamental Rights question
      when the retrieved excerpts do not establish that conclusion?
    - Did I avoid unsupported conclusions?

    Return ONLY the final answer.
"""

# ============================================================
# LOCAL OLLAMA
# ============================================================

def generate_with_local_llm(
    prompt: str
) -> str:
    """
    Generate using the local Ollama model.
    """

    import ollama

    response = ollama.chat(
        model=LOCAL_LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response["message"]["content"]
        .strip()
    )


# ============================================================
# GEMINI
# ============================================================

def generate_with_gemini(
    prompt: str
) -> str:
    """
    Generate using Gemini when configured.
    """

    if gemini_client is None:

        raise RuntimeError(
            "Gemini client is not configured."
        )

    interaction = (
        gemini_client
        .interactions.create(
            model=GEMINI_MODEL,
            input=prompt
        )
    )

    return (
        interaction.output_text
        .strip()
    )


# ============================================================
# MAIN GROUNDED GENERATION
# ============================================================

def generate_grounded_answer(
    question: str,
    retrieved_chunks: list
) -> str:
    """
    Generate an answer from retrieved evidence.

    Local Qwen is the default provider.
    Gemini can be selected through .env.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_chunks:

        return (
            "No relevant evidence was found "
            "in the indexed judgment."
        )

    # --------------------------------------------------------
    # Build evidence
    # --------------------------------------------------------

    evidence = build_evidence_context(
        retrieved_chunks
    )

    # --------------------------------------------------------
    # Build deterministic legal facts
    # --------------------------------------------------------

    structured_facts = (
        build_structured_evidence(
            retrieved_chunks
        )
    )

    # --------------------------------------------------------
    # Build grounded prompt
    # --------------------------------------------------------

    prompt = build_grounded_prompt(
        question=question,
        evidence=evidence,
        structured_facts=structured_facts
    )

    # --------------------------------------------------------
    # Local model
    # --------------------------------------------------------

    if LLM_PROVIDER == "local":

        return generate_with_local_llm(
            prompt
        )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    if LLM_PROVIDER == "gemini":

        return generate_with_gemini(
            prompt
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: "
        f"{LLM_PROVIDER}. "
        f"Use 'local' or 'gemini'."
    )
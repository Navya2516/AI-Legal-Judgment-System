import json
import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


client = genai.Client(
    api_key=API_KEY,
    http_options={"timeout": 30000}
)

MODEL_NAME = "gemini-3.6-flash"

# ============================================================
# EVIDENCE SELECTION
# ============================================================

def select_provision_evidence(
    provision_data: dict,
    max_evidence: int = 8
) -> list:
    """
    Select a limited number of evidence records.

    We do not send every extracted occurrence to Gemini.
    This keeps the prompt manageable while preserving
    representative evidence.
    """

    evidence = provision_data.get(
        "evidence",
        []
    )

    if not evidence:
        return []

    # Prefer high-confidence evidence.
    high_confidence = [
        item
        for item in evidence
        if item.get("confidence") == "high"
    ]

    medium_confidence = [
        item
        for item in evidence
        if item.get("confidence") == "medium"
    ]

    low_confidence = [
        item
        for item in evidence
        if item.get("confidence") == "low"
    ]

    selected = (
        high_confidence
        + medium_confidence
        + low_confidence
    )

    # Remove duplicate page/text combinations.
    unique = []

    seen = set()

    for item in selected:

        page = item.get("page")
        text = item.get("text", "").strip()

        key = (
            page,
            text
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(item)

        if len(unique) >= max_evidence:
            break

    return unique


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

def build_continuation_evidence_context(
    ranked_provisions: list,
    validated_provisions: dict,
    max_evidence_per_provision: int = 8
) -> str:
    """
    Build evidence context for the selected top provisions.
    """

    context_parts = []

    for rank, ranked_item in enumerate(
        ranked_provisions,
        start=1
    ):

        provision = ranked_item.get(
            "provision"
        )

        category = ranked_item.get(
            "category"
        )

        # ----------------------------------------------------
        # Locate original provision data
        # ----------------------------------------------------

        if category == "Article":

            provision_data = (
                validated_provisions
                .get("articles", {})
                .get(provision)
            )

        elif category == "Section":

            provision_data = (
                validated_provisions
                .get("sections", {})
                .get(provision)
            )

        elif category == (
            "Constitutional Amendment"
        ):

            provision_data = (
                validated_provisions
                .get(
                    "constitutional_amendments",
                    {}
                )
                .get(provision)
            )

        else:
            provision_data = None

        if not provision_data:
            continue

        selected_evidence = (
            select_provision_evidence(
                provision_data,
                max_evidence=max_evidence_per_provision
            )
        )

        if not selected_evidence:
            continue

        section = []

        section.append(
            f"PROVISION {rank}"
        )

        section.append(
            f"Name: {provision}"
        )

        section.append(
            f"Category: {category}"
        )

        section.append(
            f"Relevance Score: "
            f"{ranked_item.get('relevance_score')}"
        )

        section.append(
            f"Ranking Confidence: "
            f"{ranked_item.get('confidence')}"
        )

        section.append(
            ""
        )

        for evidence_number, item in enumerate(
            selected_evidence,
            start=1
        ):

            page = item.get(
                "page",
                "Unknown"
            )

            confidence = item.get(
                "confidence",
                "Unknown"
            )

            text = item.get(
                "text",
                ""
            ).strip()

            section.append(
                f"SOURCE {evidence_number}"
            )

            section.append(
                f"Page: {page}"
            )

            section.append(
                f"Confidence: {confidence}"
            )

            section.append(
                text
            )

            section.append(
                ""
            )

        context_parts.append(
            "\n".join(section)
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# PROMPT
# ============================================================

def build_case_continuation_prompt(
    ranked_provisions: list,
    validated_provisions: dict
) -> str:
    """
    Build a grounded prompt for identifying
    potential continuation-related legal issues.
    """

    evidence_context = (
        build_continuation_evidence_context(
            ranked_provisions,
            validated_provisions
        )
    )

    provision_names = [
        item.get("provision")
        for item in ranked_provisions
    ]

    prompt = f"""
You are an evidence-grounded AI assistant
for explaining Indian court judgments.

You are given selected legal provisions and
evidence extracted from an uploaded judgment.

Your task is to identify POTENTIAL LEGAL
PROVISIONS AND ISSUES THAT MAY BE RELEVANT
TO CONTINUING RESEARCH OR FURTHER PROCEEDINGS
RELATED TO THE CASE.

IMPORTANT SAFETY AND ACCURACY RULES:

1. Use ONLY the supplied judgment evidence.

2. Do NOT use outside legal knowledge to introduce
   unsupported claims.

3. Do NOT predict the result of any future case,
   hearing, appeal, petition, or proceeding.

4. Do NOT provide personalized legal advice.

5. Do NOT tell the user what they should argue.

6. Do NOT recommend a legal strategy.

7. Do NOT claim that a provision MUST be used.

8. Do NOT claim that a court WILL decide something.

9. Distinguish clearly between:
   - what the judgment actually discusses
   - what issue may require further research

10. Every important claim must have a page reference.

11. If the evidence is insufficient, explicitly say so.

12. "Potential continuation insight" means a research
    lead derived from the judgment, NOT a prediction
    or legal recommendation.

------------------------------------------------------------
SELECTED PROVISIONS
------------------------------------------------------------

{json.dumps(
    provision_names,
    indent=2,
    ensure_ascii=False
)}

------------------------------------------------------------
JUDGMENT EVIDENCE
------------------------------------------------------------

{evidence_context}

------------------------------------------------------------
TASK
------------------------------------------------------------

Return ONLY valid JSON.

Use exactly this structure:

{{
    "potential_legal_provisions": [
        {{
            "provision": "Article 368",
            "why_relevant": "...",
            "judgment_discussion": "...",
            "potential_continuation_issue": "...",
            "evidence": [
                {{
                    "page": 1,
                    "support": "..."
                }}
            ],
            "confidence": "high",
            "limitation": "..."
        }}
    ],

    "issues_for_continuation": [
        {{
            "issue": "...",
            "description": "...",
            "evidence": [
                {{
                    "page": 1,
                    "support": "..."
                }}
            ],
            "confidence": "high",
            "limitation": "..."
        }}
    ],

    "related_provisions": [
        {{
            "provision": "Article 13",
            "relationship": "...",
            "evidence": [
                {{
                    "page": 1,
                    "support": "..."
                }}
            ]
        }}
    ],

    "overall_limitation": "...",

    "disclaimer":
        "This output is an evidence-grounded research summary generated from the uploaded judgment. It is not legal advice and should be verified by a qualified legal professional."
}}

------------------------------------------------------------
QUALITY REQUIREMENTS
------------------------------------------------------------

For potential_legal_provisions:

- Include only provisions supported by the supplied evidence.
- Explain why each provision appears relevant to the judgment.
- Identify a potential continuation issue only when the
  evidence supports such a research lead.
- Include page references.

For issues_for_continuation:

- Describe unresolved, continuing, related, or potentially
  research-relevant issues visible in the judgment evidence.
- Do not invent issues.
- Do not predict future outcomes.
- Do not prescribe legal action.
- Include page references.

For related_provisions:

- Include relationships supported by the supplied evidence.
- Do not invent legal relationships from outside knowledge.

Confidence must be exactly one of:

high
medium
low

The final answer must contain valid JSON only.
"""

    return prompt


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_case_continuation_response(
    response_text: str
) -> dict:
    """
    Convert Gemini response into a Python dictionary.
    """

    cleaned = response_text.strip()

    if cleaned.startswith("```json"):

        cleaned = cleaned[
            len("```json"):
        ]

    elif cleaned.startswith("```"):

        cleaned = cleaned[
            len("```"):
        ]

    if cleaned.endswith("```"):

        cleaned = cleaned[
            :-len("```")
        ]

    cleaned = cleaned.strip()

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if (
            start != -1
            and end != -1
        ):

            candidate = cleaned[
                start:end + 1
            ]

            try:

                return json.loads(
                    candidate
                )

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Gemini returned an invalid JSON response."
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_case_continuation_response(
    result: dict
) -> dict:
    """
    Validate the structure returned by Gemini.
    """

    required_fields = [
        "potential_legal_provisions",
        "issues_for_continuation",
        "related_provisions",
        "overall_limitation",
        "disclaimer"
    ]

    for field in required_fields:

        if field not in result:

            raise ValueError(
                f"Missing field: {field}"
            )

    if not isinstance(
        result["potential_legal_provisions"],
        list
    ):

        raise ValueError(
            "potential_legal_provisions must be a list."
        )

    if not isinstance(
        result["issues_for_continuation"],
        list
    ):

        raise ValueError(
            "issues_for_continuation must be a list."
        )

    if not isinstance(
        result["related_provisions"],
        list
    ):

        raise ValueError(
            "related_provisions must be a list."
        )

    allowed_confidence = {
        "high",
        "medium",
        "low"
    }

    # --------------------------------------------------------
    # Validate provision insights
    # --------------------------------------------------------

    for item in result[
        "potential_legal_provisions"
    ]:

        if not isinstance(
            item,
            dict
        ):
            continue

        if item.get(
            "confidence"
        ) not in allowed_confidence:

            item[
                "confidence"
            ] = "low"

        if not isinstance(
            item.get("evidence", []),
            list
        ):

            item["evidence"] = []

    # --------------------------------------------------------
    # Validate continuation issues
    # --------------------------------------------------------

    for item in result[
        "issues_for_continuation"
    ]:

        if not isinstance(
            item,
            dict
        ):
            continue

        if item.get(
            "confidence"
        ) not in allowed_confidence:

            item[
                "confidence"
            ] = "low"

        if not isinstance(
            item.get("evidence", []),
            list
        ):

            item["evidence"] = []

    # --------------------------------------------------------
    # Validate related provisions
    # --------------------------------------------------------

    for item in result[
        "related_provisions"
    ]:

        if not isinstance(
            item,
            dict
        ):
            continue

        if not isinstance(
            item.get("evidence", []),
            list
        ):

            item["evidence"] = []

    return result


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_case_continuation_insights(
    ranked_provisions: list,
    validated_provisions: dict
) -> dict:
    """
    Generate evidence-grounded continuation insights.
    """

    if not ranked_provisions:

        raise ValueError(
            "No ranked provisions were supplied."
        )

    prompt = (
        build_case_continuation_prompt(
            ranked_provisions,
            validated_provisions
        )
    )

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    response_text = (
        interaction.output_text
    )

    if not response_text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    result = (
        parse_case_continuation_response(
            response_text
        )
    )

    result = (
        validate_case_continuation_response(
            result
        )
    )

    return result
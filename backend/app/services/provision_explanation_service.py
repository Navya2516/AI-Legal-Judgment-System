import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "local"
).lower()


# ============================================================
# PROVISION NORMALIZATION
# ============================================================

def normalize_provision(provision: str) -> str:
    """
    Normalize the requested legal provision.
    """

    if not provision:
        return ""

    provision = str(provision).strip()

    provision = re.sub(
        r"\s+",
        " ",
        provision
    )

    return provision


def get_provision_variants(provision: str) -> list:
    """
    Generate searchable variants for a legal provision.
    """

    provision = normalize_provision(
        provision
    )

    variants = []

    if not provision:
        return variants

    lower = provision.lower()

    variants.append(
        lower
    )

    # --------------------------------------------------------
    # Article
    # --------------------------------------------------------

    match = re.search(
        r"article\s+([0-9]+[a-z]?(?:\([0-9a-z]+\))*)",
        lower
    )

    if match:

        number = match.group(1)

        variants.extend(
            [
                f"article {number}",
                f"article {number} of the constitution",
                f"article {number} constitution",
            ]
        )

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    match = re.search(
        r"section\s+([0-9]+[a-z]?(?:\([0-9a-z]+\))*)",
        lower
    )

    if match:

        number = match.group(1)

        variants.extend(
            [
                f"section {number}",
                f"section {number} of the act",
            ]
        )

    # --------------------------------------------------------
    # Constitutional amendment
    # --------------------------------------------------------

    match = re.search(
        r"(\d+)(?:st|nd|rd|th)?\s+amendment",
        lower
    )

    if match:

        number = match.group(1)

        variants.extend(
            [
                f"{number}th amendment",
                f"{number} amendment",
                f"constitutional amendment",
            ]
        )

    return list(
        dict.fromkeys(variants)
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text) -> str:
    """
    Clean extracted evidence text.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def extract_page(item):
    """
    Extract page number from an evidence record.
    """

    if not isinstance(item, dict):
        return None

    page = item.get(
        "page",
        item.get(
            "page_number",
            item.get(
                "page_no",
                item.get("source_page")
            )
        )
    )

    try:

        return int(page)

    except (
        ValueError,
        TypeError
    ):

        return None


def extract_text(item) -> str:
    """
    Extract evidence text from different possible
    evidence dictionary structures.
    """

    if isinstance(item, str):
        return clean_text(item)

    if not isinstance(item, dict):
        return ""

    text = item.get(
        "text",
        item.get(
            "content",
            item.get(
                "chunk",
                item.get(
                    "page_text",
                    item.get(
                        "evidence",
                        ""
                    )
                )
            )
        )
    )

    return clean_text(text)


def extract_score(item):
    """
    Extract similarity/relevance score when available.
    """

    if not isinstance(item, dict):
        return 0.0

    score = item.get(
        "score",
        item.get(
            "similarity",
            item.get(
                "relevance_score",
                0.0
            )
        )
    )

    try:
        return float(score)

    except (
        ValueError,
        TypeError
    ):

        return 0.0


# ============================================================
# CHECK PROVISION MENTION
# ============================================================

def text_mentions_provision(
    text: str,
    provision: str
) -> bool:

    text = clean_text(
        text
    ).lower()

    if not text:
        return False

    variants = get_provision_variants(
        provision
    )

    for variant in variants:

        if variant and variant in text:
            return True

    return False


# ============================================================
# COLLECT ALL EVIDENCE
# ============================================================

def collect_evidence(
    provision_data: dict
) -> list:
    """
    Extract evidence from the provision service output.

    Supports:
    - evidence
    - records
    - results
    - passages
    - pages
    """

    if not isinstance(
        provision_data,
        dict
    ):
        return []

    candidates = []

    possible_keys = [
        "evidence",
        "records",
        "results",
        "passages",
        "chunks"
    ]

    for key in possible_keys:

        value = provision_data.get(
            key
        )

        if isinstance(
            value,
            list
        ):

            candidates.extend(
                value
            )

    # --------------------------------------------------------
    # Some service outputs may store page information
    # directly.
    # --------------------------------------------------------

    pages = provision_data.get(
        "pages"
    )

    if isinstance(
        pages,
        list
    ):

        for page in pages:

            if isinstance(
                page,
                dict
            ):

                candidates.append(
                    page
                )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = []

    seen = set()

    for item in candidates:

        page = extract_page(
            item
        )

        text = extract_text(
            item
        )

        if not text:
            continue

        key = (
            page,
            text[:500]
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            {
                "page": page,
                "text": text,
                "score": extract_score(item),
                "original": item
            }
        )

    return unique


# ============================================================
# SELECT PROVISION-SPECIFIC EVIDENCE
# ============================================================

def select_relevant_evidence(
    provision: str,
    provision_data: dict,
    max_items: int = 8
) -> list:
    """
    Select evidence that actually relates to the requested
    provision.

    This is deliberately deterministic.
    """

    all_evidence = collect_evidence(
        provision_data
    )

    if not all_evidence:
        return []

    direct = []
    indirect = []

    for item in all_evidence:

        text = item["text"]

        if text_mentions_provision(
            text,
            provision
        ):

            direct.append(
                item
            )

        else:

            indirect.append(
                item
            )

    # --------------------------------------------------------
    # Direct evidence gets highest priority.
    # --------------------------------------------------------

    direct.sort(
        key=lambda x: (
            x["score"],
            x["page"] or 0
        ),
        reverse=True
    )

    indirect.sort(
        key=lambda x: (
            x["score"],
            x["page"] or 0
        ),
        reverse=True
    )

    selected = direct[:max_items]

    # --------------------------------------------------------
    # Only use indirect evidence if we do not have enough
    # direct evidence.
    # --------------------------------------------------------

    if len(selected) < max_items:

        remaining = max_items - len(
            selected
        )

        selected.extend(
            indirect[:remaining]
        )

    return selected


# ============================================================
# CLASSIFY EVIDENCE
# ============================================================

def classify_evidence(
    text: str
) -> str:
    """
    Conservative evidence classification.

    This does not infer legal conclusions.
    """

    lower = text.lower()

    # --------------------------------------------------------
    # Party arguments
    # --------------------------------------------------------

    argument_markers = [
        "it was argued",
        "it was urged",
        "it was submitted",
        "it was contended",
        "counsel contended",
        "counsel submitted",
        "on behalf of the petitioner",
        "on behalf of the petitioners",
        "on behalf of the respondent",
        "on behalf of the respondents",
        "the petitioner submitted",
        "the petitioners submitted",
        "the respondents contended",
        "the respondents submitted",
    ]

    for marker in argument_markers:

        if marker in lower:

            return "party_argument"

    # --------------------------------------------------------
    # Judicial reasoning
    # --------------------------------------------------------

    reasoning_markers = [
        "we are unable to accept",
        "we agree",
        "we hold",
        "we are of the opinion",
        "in our opinion",
        "the court holds",
        "the court held",
        "the majority held",
        "it is clear",
        "we conclude",
        "we accordingly hold",
        "for these reasons",
        "in view of the above",
    ]

    for marker in reasoning_markers:

        if marker in lower:

            return "judicial_reasoning"

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    decision_markers = [
        "the petition is dismissed",
        "the petitions are dismissed",
        "the petition is allowed",
        "the petitions are allowed",
        "the appeal is dismissed",
        "the appeals are dismissed",
        "the appeal is allowed",
        "the appeals are allowed",
        "accordingly",
        "in the result",
        "we accordingly",
    ]

    for marker in decision_markers:

        if marker in lower:

            return "final_decision"

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "court_discussion"


# ============================================================
# BUILD EVIDENCE RECORDS
# ============================================================

def build_evidence_records(
    evidence: list,
    provision: str
) -> list:

    records = []

    for item in evidence:

        page = item.get(
            "page"
        )

        text = clean_text(
            item.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        evidence_type = classify_evidence(
            text
        )

        records.append(
            {
                "page": page,
                "text": text,
                "type": evidence_type
            }
        )

    return records


# ============================================================
# BUILD PAGE CITATION
# ============================================================

def page_citation(page) -> str:

    if page is None:
        return ""

    return f"[Page {page}]"


# ============================================================
# BUILD EVIDENCE SENTENCE
# ============================================================

def build_evidence_sentence(
    evidence_item: dict,
    provision: str
) -> str:

    text = clean_text(
        evidence_item.get(
            "text",
            ""
        )
    )

    page = evidence_item.get(
        "page"
    )

    citation = page_citation(
        page
    )

    if not text:
        return ""

    # --------------------------------------------------------
    # Do not rewrite the evidence into unsupported legal
    # conclusions. Keep it close to the original evidence.
    # --------------------------------------------------------

    if citation:

        return (
            f"The judgment evidence states: "
            f"\"{text}\" {citation}"
        )

    return (
        f"The judgment evidence states: "
        f"\"{text}\""
    )


# ============================================================
# BUILD DETERMINISTIC EXPLANATION
# ============================================================

def build_grounded_explanation(
    provision: str,
    selected_evidence: list
) -> dict:
    """
    Generate the core explanation without relying on an LLM.

    This guarantees that the output is grounded in the
    retrieved judgment evidence.
    """

    provision = normalize_provision(
        provision
    )

    records = build_evidence_records(
        selected_evidence,
        provision
    )

    direct_records = []

    for record in records:

        if text_mentions_provision(
            record["text"],
            provision
        ):

            direct_records.append(
                record
            )

    # ========================================================
    # NO DIRECT EVIDENCE
    # ========================================================

    if not direct_records:

        return {
            "provision": provision,

            "what_the_provision_is_discussed_as": (
                f"The supplied evidence does not directly "
                f"identify {provision}. No provision-specific "
                f"explanation can be safely generated from "
                f"the retrieved evidence."
            ),

            "why_it_appears_in_the_judgment": (
                f"The retrieved evidence is insufficient "
                f"to establish why {provision} appears in "
                f"the judgment."
            ),

            "court_discussion": (
                f"The supplied evidence does not provide "
                f"sufficient direct material to describe "
                f"the court's discussion of {provision}."
            ),

            "evidence": [],

            "confidence": "low",

            "limitation": (
                f"The retrieved evidence does not directly "
                f"support an explanation of {provision}."
            )
        }

    # ========================================================
    # DIRECT EVIDENCE EXISTS
    # ========================================================

    first = direct_records[0]

    page = first.get(
        "page"
    )

    citation = page_citation(
        page
    )

    # --------------------------------------------------------
    # Determine whether evidence is mostly party argument.
    # --------------------------------------------------------

    argument_count = sum(
        1
        for item in direct_records
        if item["type"] == "party_argument"
    )

    judicial_count = sum(
        1
        for item in direct_records
        if item["type"] in {
            "court_discussion",
            "judicial_reasoning",
            "final_decision"
        }
    )

    # ========================================================
    # WHAT IS IT DISCUSSED AS?
    # ========================================================

    if judicial_count > 0:

        what_discussed = (
            f"{provision} is directly discussed in "
            f"the supplied judgment evidence. "
            f"{citation}"
        )

    else:

        what_discussed = (
            f"{provision} is directly referred to in "
            f"the supplied judgment evidence, primarily "
            f"in the context of the recorded submissions "
            f"or arguments. {citation}"
        )

    # ========================================================
    # WHY DOES IT APPEAR?
    # ========================================================

    if argument_count > 0 and judicial_count == 0:

        why_appears = (
            f"The supplied evidence shows that "
            f"{provision} appears in connection with "
            f"a party's submission or argument. "
            f"The evidence does not by itself establish "
            f"a broader judicial reason. {citation}"
        )

    else:

        why_appears = (
            f"The supplied evidence shows that "
            f"{provision} forms part of the legal discussion "
            f"in the judgment. {citation}"
        )

    # ========================================================
    # COURT DISCUSSION
    # ========================================================

    court_records = [
        item
        for item in direct_records
        if item["type"] in {
            "court_discussion",
            "judicial_reasoning",
            "final_decision"
        }
    ]

    if court_records:

        court_item = court_records[0]

        court_text = clean_text(
            court_item["text"]
        )

        court_page = court_item.get(
            "page"
        )

        court_citation = page_citation(
            court_page
        )

        court_discussion = (
            f"The supplied judgment evidence directly "
            f"discusses {provision}: "
            f"\"{court_text}\" "
            f"{court_citation}"
        )

    else:

        court_discussion = (
            f"The supplied evidence directly mentions "
            f"{provision}, but the retrieved material is "
            f"primarily associated with party submissions "
            f"and does not establish a separate judicial "
            f"conclusion about the provision. "
            f"{citation}"
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if (
        len(direct_records) >= 2
        and judicial_count > 0
    ):

        confidence = "high"

    elif judicial_count > 0:

        confidence = "medium"

    else:

        confidence = "medium"

    # ========================================================
    # LIMITATION
    # ========================================================

    if (
        argument_count > 0
        and judicial_count == 0
    ):

        limitation = (
            f"The retrieved evidence directly refers to "
            f"{provision}, but it primarily reflects "
            f"party submissions or arguments. It does not "
            f"establish a final judicial conclusion."
        )

    elif len(direct_records) == 1:

        limitation = (
            f"The explanation is based on limited "
            f"provision-specific evidence and should not "
            f"be treated as a complete account of the "
            f"court's treatment of {provision}."
        )

    else:

        limitation = (
            f"The explanation is limited to the "
            f"provision-specific evidence retrieved from "
            f"the judgment."
        )

    return {
        "provision": provision,

        "what_the_provision_is_discussed_as":
            what_discussed,

        "why_it_appears_in_the_judgment":
            why_appears,

        "court_discussion":
            court_discussion,

        "evidence":
            direct_records,

        "confidence":
            confidence,

        "limitation":
            limitation
    }


# ============================================================
# OPTIONAL LLM ENHANCEMENT
# ============================================================

def optionally_enhance_with_llm(
    explanation: dict
) -> dict:
    """
    Currently disabled by default.

    The deterministic explanation is safer for the legal
    provision module. The LLM is already used elsewhere
    in the project for summary and RAG generation.

    Set:
        PROVISION_LLM_ENHANCEMENT=true

    only if you want to experiment with LLM rewriting.
    """

    enabled = os.getenv(
        "PROVISION_LLM_ENHANCEMENT",
        "false"
    ).lower()

    if enabled != "true":
        return explanation

    # --------------------------------------------------------
    # Import only when explicitly enabled.
    # --------------------------------------------------------

    try:

        from backend.app.services.llm_service import (
            generate_with_local_llm
        )

    except Exception:

        return explanation

    evidence_json = json.dumps(
        explanation,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
Rewrite the following legal provision explanation
for readability.

IMPORTANT:

Do NOT add facts.

Do NOT add legal knowledge.

Do NOT remove evidence.

Do NOT remove page citations.

Do NOT change the requested provision.

Do NOT change the evidence array.

Do NOT change confidence.

Do NOT change limitation.

Return exactly the same JSON structure.

Requested provision:
{explanation["provision"]}

Current grounded explanation:

{evidence_json}
"""

    try:

        raw = generate_with_local_llm(
            prompt
        )

        if not raw:
            return explanation

        raw = raw.strip()

        if raw.startswith(
            "```json"
        ):

            raw = raw[
                len("```json"):
            ]

        if raw.endswith(
            "```"
        ):

            raw = raw[
                :-3
            ]

        candidate = json.loads(
            raw.strip()
        )

        # ----------------------------------------------------
        # Only accept enhancement if the critical fields
        # remain intact.
        # ----------------------------------------------------

        if candidate.get(
            "provision"
        ) != explanation.get(
            "provision"
        ):

            return explanation

        if candidate.get(
            "evidence"
        ) != explanation.get(
            "evidence"
        ):

            return explanation

        if candidate.get(
            "confidence"
        ) != explanation.get(
            "confidence"
        ):

            return explanation

        if candidate.get(
            "limitation"
        ) != explanation.get(
            "limitation"
        ):

            return explanation

        return candidate

    except Exception:

        return explanation


# ============================================================
# VALIDATE FINAL EXPLANATION
# ============================================================

def validate_provision_response(
    result: dict,
    provision: str
) -> dict:

    provision = normalize_provision(
        provision
    )

    required_fields = [
        "provision",
        "what_the_provision_is_discussed_as",
        "why_it_appears_in_the_judgment",
        "court_discussion",
        "evidence",
        "confidence",
        "limitation"
    ]

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    for field in required_fields:

        if field not in result:

            raise ValueError(
                f"Missing required field: {field}"
            )

    # --------------------------------------------------------
    # Exact provision
    # --------------------------------------------------------

    if normalize_provision(
        result["provision"]
    ).lower() != provision.lower():

        raise ValueError(
            "Generated explanation refers to a different "
            "legal provision."
        )

    result["provision"] = provision

    # --------------------------------------------------------
    # Text fields
    # --------------------------------------------------------

    text_fields = [
        "what_the_provision_is_discussed_as",
        "why_it_appears_in_the_judgment",
        "court_discussion",
        "limitation"
    ]

    for field in text_fields:

        if not isinstance(
            result[field],
            str
        ):

            raise ValueError(
                f"{field} must be a string."
            )

        if not result[field].strip():

            raise ValueError(
                f"{field} cannot be empty."
            )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    if not isinstance(
        result["evidence"],
        list
    ):

        raise ValueError(
            "evidence must be a list."
        )

    validated_evidence = []

    allowed_types = {
        "court_discussion",
        "party_argument",
        "judicial_reasoning",
        "final_decision",
        "reference"
    }

    for item in result["evidence"]:

        if not isinstance(
            item,
            dict
        ):

            raise ValueError(
                "Evidence item must be an object."
            )

        page = item.get(
            "page"
        )

        text = clean_text(
            item.get(
                "text",
                ""
            )
        )

        evidence_type = str(
            item.get(
                "type",
                ""
            )
        ).lower().strip()

        try:

            page = int(
                page
            )

        except (
            ValueError,
            TypeError
        ):

            raise ValueError(
                "Evidence page must be an integer."
            )

        if page <= 0:

            raise ValueError(
                "Evidence page must be greater than zero."
            )

        if not text:

            raise ValueError(
                "Evidence text cannot be empty."
            )

        if evidence_type not in allowed_types:

            raise ValueError(
                f"Invalid evidence type: "
                f"{evidence_type}"
            )

        validated_evidence.append(
            {
                "page": page,
                "text": text,
                "type": evidence_type
            }
        )

    result["evidence"] = validated_evidence

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = str(
        result["confidence"]
    ).lower().strip()

    if confidence not in {
        "high",
        "medium",
        "low"
    }:

        raise ValueError(
            "confidence must be high, medium, or low."
        )

    result["confidence"] = confidence

    # --------------------------------------------------------
    # Evidence requirement
    # --------------------------------------------------------

    if not validated_evidence:

        if confidence == "high":

            result["confidence"] = "low"

        result["limitation"] = (
            result["limitation"].strip()
            + " No direct evidence record was retrieved."
        )

    # --------------------------------------------------------
    # Check evidence actually relates to provision
    # --------------------------------------------------------

    relevant_count = 0

    for item in validated_evidence:

        if text_mentions_provision(
            item["text"],
            provision
        ):

            relevant_count += 1

    if relevant_count == 0:

        result["confidence"] = "low"

        result["limitation"] = (
            result["limitation"].strip()
            + " The evidence records do not directly "
              "mention the requested provision."
        )

    # --------------------------------------------------------
    # Reject obvious unsupported speculation
    # --------------------------------------------------------

    combined = " ".join(
        [
            result[
                "what_the_provision_is_discussed_as"
            ],
            result[
                "why_it_appears_in_the_judgment"
            ],
            result[
                "court_discussion"
            ]
        ]
    ).lower()

    forbidden_phrases = [
        "potentially indicating",
        "may indicate that",
        "likely means",
        "probably means",
        "it can be inferred",
        "this suggests that",
        "might indicate"
    ]

    for phrase in forbidden_phrases:

        if phrase in combined:

            raise ValueError(
                "Generated explanation contains "
                "unsupported speculative language."
            )

    return result


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_provision_explanation(
    provision: str,
    provision_data: dict
) -> dict:
    """
    Generate an evidence-grounded explanation for a
    specific legal provision.

    The core explanation is deterministic and grounded
    directly in retrieved judgment evidence.
    """

    if not provision:

        raise ValueError(
            "Provision is required."
        )

    if not isinstance(
        provision_data,
        dict
    ):

        raise ValueError(
            "Provision data must be a dictionary."
        )

    provision = normalize_provision(
        provision
    )

    # ========================================================
    # STEP 1 — Retrieve relevant evidence
    # ========================================================

    selected_evidence = select_relevant_evidence(
        provision,
        provision_data,
        max_items=8
    )

    # ========================================================
    # STEP 2 — Build deterministic grounded explanation
    # ========================================================

    explanation = build_grounded_explanation(
        provision,
        selected_evidence
    )

    # ========================================================
    # STEP 3 — Optional LLM readability enhancement
    #
    # Disabled by default.
    # ========================================================

    explanation = optionally_enhance_with_llm(
        explanation
    )

    # ========================================================
    # STEP 4 — Final validation
    # ========================================================

    explanation = validate_provision_response(
        explanation,
        provision
    )

    return explanation
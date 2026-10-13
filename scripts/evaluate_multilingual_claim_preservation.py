import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_summaries"
    / f"{DOCUMENT_ID}_summary.json"
)

MULTILINGUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multilingual_explanations"
    / f"{DOCUMENT_ID}_tamil.json"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "multilingual_claim_preservation_results.json"
)

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# PAGE CITATION EXTRACTION
# ============================================================

def extract_page_citations(text):
    return [
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            text,
            flags=re.IGNORECASE
        )
    ]


# ============================================================
# REMOVE CITATIONS
# ============================================================

def remove_citations(text):

    text = re.sub(
        r"\[Page\s+\d+\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CLAIM SPLITTING
# ============================================================

def split_claims(text):

    text = text.replace(
        "\n",
        " "
    )

    sentences = re.split(
        r"(?<=[.!?\u0964\u0965\u0be4])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 20
    ]


# ============================================================
# BUILD ENGLISH CLAIMS
# ============================================================


def build_english_claims(summary):

    # The summary may store the generated explanation
    # inside a nested "summary" or "explanation" object.
    possible_locations = [
        summary,
        summary.get("summary", {}),
        summary.get("explanation", {})
    ]

    text = ""

    for location in possible_locations:

        if not isinstance(location, dict):
            continue

        candidate = location.get(
            "simplified_explanation",
            ""
        )

        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            break

    if not text:
        raise ValueError(
            "English simplified_explanation not found "
            "in the supplied summary JSON."
        )

    return split_claims(text)


# ============================================================
# BUILD TAMIL CLAIMS
# ============================================================

def build_tamil_claims(multilingual):

    explanation = multilingual.get(
        "explanation",
        {}
    )

    text = explanation.get(
        "simplified_explanation",
        ""
    )

    if not text:
        raise ValueError(
            "Tamil simplified_explanation not found."
        )

    return split_claims(text)


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(response_text):

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
        return json.loads(cleaned)

    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:

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
            "Gemini returned invalid JSON."
        )


# ============================================================
# BUILD CLAIM PRESERVATION PROMPT
# ============================================================

def build_claim_prompt(
    english_claim,
    tamil_claim,
    claim_number
):

    prompt = f"""
You are evaluating the faithfulness of a multilingual
legal judgment explanation.

The English sentence is the SOURCE CLAIM.

The Tamil sentence is the TRANSLATED CLAIM.

Your task is NOT to judge grammar or writing quality.

Your task is to determine whether the Tamil sentence
preserves the legal meaning of the English sentence.

Evaluate these dimensions:

1. SUBJECT
   Does the Tamil claim preserve who or what the claim
   is about?

2. LEGAL_PROVISION
   Are important legal provisions, Articles, Amendments,
   doctrines, or legal concepts preserved?

3. ACTION_OR_POWER
   Is the action, authority, power, prohibition,
   permission, or legal relationship preserved?

4. LIMITATION_OR_CONDITION
   If the English claim contains a limitation,
   exception, condition, or qualification, is it preserved?

5. CONCLUSION
   Is the resulting legal conclusion or effect preserved?

IMPORTANT:

- Do not penalize the Tamil sentence merely because
  wording and sentence structure are different.
- Do not require literal translation.
- Synonyms and natural Tamil phrasing are acceptable.
- Focus on legal meaning.
- Do not use outside legal knowledge to add facts.
- If an English claim is simplified, judge preservation
  of that simplified meaning.
- A missing important legal limitation is a serious error.
- A changed legal outcome is a serious error.

SOURCE CLAIM:

{remove_citations(english_claim)}

TRANSLATED TAMIL CLAIM:

{remove_citations(tamil_claim)}

Return ONLY this JSON structure:

{{
    "claim_number": {claim_number},
    "subject_preserved": true,
    "legal_provision_preserved": true,
    "action_or_power_preserved": true,
    "limitation_or_condition_preserved": true,
    "conclusion_preserved": true,
    "meaning_preserved": true,
    "severity": "none",
    "reason": "Brief explanation."
}}

Allowed values for severity:

"none"
"minor"
"major"

Use:

- none = meaning is faithfully preserved
- minor = small loss or wording difference that does
  not materially change the legal meaning
- major = important legal meaning, limitation,
  provision, action, or conclusion is changed,
  omitted, or contradicted

Return ONLY JSON.
"""

    return prompt


# ============================================================
# EVALUATE ONE CLAIM
# ============================================================

def evaluate_claim(
    english_claim,
    tamil_claim,
    claim_number
):

    prompt = build_claim_prompt(
        english_claim,
        tamil_claim,
        claim_number
    )

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    response_text = interaction.output_text

    if not response_text:
        raise ValueError(
            "Gemini returned an empty evaluation."
        )

    result = parse_json_response(
        response_text
    )

    return result


# ============================================================
# VALIDATE EVALUATION
# ============================================================

def validate_result(result):

    required_fields = [
        "claim_number",
        "subject_preserved",
        "legal_provision_preserved",
        "action_or_power_preserved",
        "limitation_or_condition_preserved",
        "conclusion_preserved",
        "meaning_preserved",
        "severity",
        "reason"
    ]

    for field in required_fields:

        if field not in result:

            raise ValueError(
                f"Missing evaluator field: {field}"
            )

    boolean_fields = [
        "subject_preserved",
        "legal_provision_preserved",
        "action_or_power_preserved",
        "limitation_or_condition_preserved",
        "conclusion_preserved",
        "meaning_preserved"
    ]

    for field in boolean_fields:

        result[field] = bool(
            result[field]
        )

    if result["severity"] not in {
        "none",
        "minor",
        "major"
    }:

        result["severity"] = "major"

    return result


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "MULTILINGUAL CLAIM PRESERVATION EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print("\nLoading English summary...")

    summary = load_json(
        SUMMARY_FILE
    )

    print(
        "Loading Tamil explanation..."
    )

    multilingual = load_json(
        MULTILINGUAL_FILE
    )

    # --------------------------------------------------------
    # Build claims
    # --------------------------------------------------------

    english_claims = build_english_claims(
        summary
    )

    tamil_claims = build_tamil_claims(
        multilingual
    )

    print(
        f"\nEnglish claims : {len(english_claims)}"
    )

    print(
        f"Tamil claims   : {len(tamil_claims)}"
    )

    # --------------------------------------------------------
    # Compare claims
    # --------------------------------------------------------

    compared_count = min(
        len(english_claims),
        len(tamil_claims)
    )

    claim_results = []

    for index in range(
        compared_count
    ):

        claim_number = index + 1

        print(
            f"\nEvaluating claim {claim_number}..."
        )

        result = evaluate_claim(
            english_claim=english_claims[index],
            tamil_claim=tamil_claims[index],
            claim_number=claim_number
        )

        result = validate_result(
            result
        )

        result[
            "english_claim"
        ] = english_claims[index]

        result[
            "tamil_claim"
        ] = tamil_claims[index]

        result[
            "english_citations"
        ] = extract_page_citations(
            english_claims[index]
        )

        result[
            "tamil_citations"
        ] = extract_page_citations(
            tamil_claims[index]
        )

        claim_results.append(
            result
        )

        print(
            f"  Meaning preserved : "
            f"{result['meaning_preserved']}"
        )

        print(
            f"  Severity          : "
            f"{result['severity']}"
        )

        print(
            f"  Reason            : "
            f"{result['reason']}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    meaning_preserved_count = sum(
        1
        for result in claim_results
        if result["meaning_preserved"]
    )

    minor_count = sum(
        1
        for result in claim_results
        if result["severity"] == "minor"
    )

    major_count = sum(
        1
        for result in claim_results
        if result["severity"] == "major"
    )

    total = len(
        claim_results
    )

    meaning_preservation_rate = (
        meaning_preserved_count / total
        if total
        else 0
    )

    # --------------------------------------------------------
    # Dimension metrics
    # --------------------------------------------------------

    dimensions = [
        "subject_preserved",
        "legal_provision_preserved",
        "action_or_power_preserved",
        "limitation_or_condition_preserved",
        "conclusion_preserved"
    ]

    dimension_scores = {}

    for dimension in dimensions:

        preserved = sum(
            1
            for result in claim_results
            if result[dimension]
        )

        dimension_scores[dimension] = (
            preserved / total
            if total
            else 0
        )

    # --------------------------------------------------------
    # Citation metrics
    # --------------------------------------------------------

    total_tamil_claims_with_citations = sum(
        1
        for result in claim_results
        if result["tamil_citations"]
    )

    citation_coverage = (
        total_tamil_claims_with_citations
        / total
        if total
        else 0
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    results = {

        "document_id":
            DOCUMENT_ID,

        "language":
            "Tamil",

        "evaluation_type":
            "llm_assisted_legal_claim_preservation",

        "evaluator_model":
            MODEL_NAME,

        "english_claim_count":
            len(english_claims),

        "tamil_claim_count":
            len(tamil_claims),

        "compared_claim_count":
            total,

        "meaning_preserved_claims":
            meaning_preserved_count,

        "meaning_preservation_rate":
            round(
                meaning_preservation_rate,
                4
            ),

        "minor_issues":
            minor_count,

        "major_issues":
            major_count,

        "dimension_scores":
            {
                key: round(
                    value,
                    4
                )
                for key, value
                in dimension_scores.items()
            },

        "tamil_claims_with_citations":
            total_tamil_claims_with_citations,

        "citation_coverage":
            round(
                citation_coverage,
                4
            ),

        "claim_results":
            claim_results,

        "evaluation_note":
            (
                "This is an LLM-assisted claim-preservation "
                "evaluation. It should be treated as an "
                "auxiliary evaluation metric and should "
                "ideally be validated with manually reviewed "
                "legal examples or expert evaluation."
            )
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"\nEnglish claims             : "
        f"{len(english_claims)}"
    )

    print(
        f"Tamil claims               : "
        f"{len(tamil_claims)}"
    )

    print(
        f"Compared claims            : "
        f"{total}"
    )

    print(
        f"Meaning preserved          : "
        f"{meaning_preserved_count}"
    )

    print(
        f"Meaning preservation       : "
        f"{meaning_preservation_rate * 100:.2f}%"
    )

    print(
        f"Minor issues               : "
        f"{minor_count}"
    )

    print(
        f"Major issues               : "
        f"{major_count}"
    )

    print(
        f"\nCitation coverage          : "
        f"{citation_coverage * 100:.2f}%"
    )

    print("\nDimension scores:")

    for dimension, score in dimension_scores.items():

        display_name = (
            dimension
            .replace("_", " ")
            .title()
        )

        print(
            f"  {display_name:<28}: "
            f"{score * 100:.2f}%"
        )

    print("\n" + "-" * 70)

    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
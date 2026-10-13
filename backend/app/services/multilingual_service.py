import json
import os
import re

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )

client = genai.Client(
    api_key=API_KEY,
    http_options={"timeout": 30000}
)

MODEL_NAME = "gemini-3.1-flash-lite-preview"

MAX_GENERATION_ATTEMPTS = 3


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = {
    "english": "English",
    "tamil": "Tamil",
    "hindi": "Hindi",
    "telugu": "Telugu",
    "malayalam": "Malayalam",
    "kannada": "Kannada"
}


# ============================================================
# BUILD SUMMARY CONTEXT
# ============================================================

def build_summary_context(summary: dict) -> str:
    """
    Convert the structured judgment summary into JSON text.
    """

    return json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# EXTRACT PAGE CITATIONS
# ============================================================

def extract_page_citations(text: str) -> list:
    """
    Extract page citations.

    Example:
        [Page 1] [Page 71] [Page 147]

    Returns a sorted list of unique page numbers.
    """

    citations = re.findall(
        r"\[Page\s+(\d+)\]",
        text,
        flags=re.IGNORECASE
    )

    return sorted(
        set(int(page) for page in citations)
    )


# ============================================================
# EXTRACT VALID SOURCE CITATIONS
# ============================================================

def extract_valid_source_citations(
    summary: dict,
    total_pages: int
) -> list:
    """
    Extract only valid page citations from the source summary.

    A citation is valid only when:

        1 <= page <= total_pages

    This is important because the generated English summary
    may itself contain an invalid citation such as [Page 2264].
    Such invalid citations must NOT be treated as authoritative
    evidence for multilingual generation.
    """

    source_text = json.dumps(
        summary,
        ensure_ascii=False
    )

    all_citations = extract_page_citations(
        source_text
    )

    valid_citations = [
        page
        for page in all_citations
        if 1 <= page <= total_pages
    ]

    return sorted(
        set(valid_citations)
    )


# ============================================================
# EXTRACT INVALID SOURCE CITATIONS
# ============================================================

def extract_invalid_source_citations(
    summary: dict,
    total_pages: int
) -> list:
    """
    Identify invalid citations already present in the
    supplied summary.
    """

    source_text = json.dumps(
        summary,
        ensure_ascii=False
    )

    all_citations = extract_page_citations(
        source_text
    )

    invalid_citations = [
        page
        for page in all_citations
        if page < 1 or page > total_pages
    ]

    return sorted(
        set(invalid_citations)
    )


# ============================================================
# EXTRACT OUTPUT PAGE CITATIONS
# ============================================================

def extract_output_page_citations(
    result: dict
) -> list:
    """
    Extract page citations from the complete multilingual
    response.
    """

    output_text = json.dumps(
        result,
        ensure_ascii=False
    )

    return extract_page_citations(
        output_text
    )


# ============================================================
# VALIDATE CITATION PRESERVATION
# ============================================================

def validate_citation_preservation(
    source_summary: dict,
    generated_result: dict,
    total_pages: int
) -> None:
    """
    Validate multilingual citation integrity.

    Important:

    Invalid citations already present in the source summary
    are NOT treated as citations that must be preserved.

    Only valid source citations are authoritative.

    Checks:

    1. Generated citations must exist in the valid source set.
    2. All valid source citations must be preserved.
    3. Generated citations must belong to the original judgment.
    """

    valid_source_citations = extract_valid_source_citations(
        source_summary,
        total_pages
    )

    generated_citations = extract_output_page_citations(
        generated_result
    )

    source_set = set(
        valid_source_citations
    )

    generated_set = set(
        generated_citations
    )

    # --------------------------------------------------------
    # CHECK 1 — INVALID GENERATED CITATIONS
    # --------------------------------------------------------

    invalid_range = sorted(
        page
        for page in generated_set
        if page < 1 or page > total_pages
    )

    if invalid_range:
        raise ValueError(
            "Multilingual generation contains page citations "
            f"outside the original judgment range "
            f"(1-{total_pages}): {invalid_range}"
        )

    # --------------------------------------------------------
    # CHECK 2 — INVENTED CITATIONS
    # --------------------------------------------------------

    invented = sorted(
        generated_set - source_set
    )

    if invented:
        raise ValueError(
            "Multilingual generation introduced page citations "
            "not present as valid citations in the source summary: "
            f"{invented}"
        )

    # --------------------------------------------------------
    # CHECK 3 — MISSING VALID SOURCE CITATIONS
    # --------------------------------------------------------

    missing = sorted(
        source_set - generated_set
    )

    if missing:
        raise ValueError(
            "Multilingual generation removed valid page citations "
            "from the source summary: "
            f"{missing}"
        )


# ============================================================
# BUILD MULTILINGUAL PROMPT
# ============================================================

def build_multilingual_prompt(
    summary: dict,
    language: str,
    total_pages: int
) -> str:
    """
    Build an evidence-preserving multilingual prompt.

    Only valid page citations from the original judgment are
    treated as authoritative.
    """

    language_key = language.lower().strip()

    if language_key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    target_language = SUPPORTED_LANGUAGES[
        language_key
    ]

    summary_context = build_summary_context(
        summary
    )

    # --------------------------------------------------------
    # Identify valid and invalid source citations
    # --------------------------------------------------------

    valid_source_citations = extract_valid_source_citations(
        summary,
        total_pages
    )

    invalid_source_citations = extract_invalid_source_citations(
        summary,
        total_pages
    )

    citation_list = ", ".join(
        f"[Page {page}]"
        for page in valid_source_citations
    )

    if not citation_list:
        citation_list = "No valid page citations available."

    invalid_citation_notice = ""

    if invalid_source_citations:
        invalid_citation_notice = f"""
IMPORTANT SOURCE CLEANUP:

The supplied summary contains the following invalid page
citation(s):

{", ".join(f"[Page {page}]" for page in invalid_source_citations)}

These citations are INVALID because the original judgment
contains only {total_pages} pages.

DO NOT copy, preserve, reproduce, or use these invalid
citations.

They must be ignored.

The valid citation list is the authoritative citation list.
"""

    prompt = f"""
You are an AI assistant that explains Indian court judgments
to ordinary citizens.

The user wants the supplied judgment summary explained in:

{target_language}

Your task is to translate and simplify the supplied summary
while preserving its meaning and valid evidence references.

============================================================
ORIGINAL JUDGMENT PAGE RANGE
============================================================

The original judgment contains exactly:

{total_pages} pages.

Therefore:

1 <= valid page number <= {total_pages}

============================================================
VALID PAGE CITATIONS
============================================================

Only the following page citations are authoritative:

{citation_list}

{invalid_citation_notice}

============================================================
CRITICAL CITATION RULE
============================================================

Page citations are IMMUTABLE EVIDENCE TOKENS.

You MUST:

1. Copy valid page citation tokens exactly.

2. Use the exact format:

   [Page X]

3. Do NOT translate page citations.

4. Do NOT change page numbers.

5. Do NOT create new page numbers.

6. Do NOT invent page citations.

7. Do NOT use page numbers outside 1-{total_pages}.

8. Do NOT use an invalid source citation.

9. Do NOT convert citations into another format.

10. Do NOT combine page numbers.

11. Do NOT split page numbers.

12. Preserve valid evidence references.

13. Never create a page number from a case number,
    section number, amendment number, paragraph number,
    date, or any other numerical information.

For example:

Correct:
[Page 128]

Correct:
[Page 128] [Page 331]

Incorrect:
[Page 2264]

Incorrect:
[Page 128-331]

Incorrect:
[பக்கம் 128]

============================================================
GENERAL RULES
============================================================

1. Use ONLY the supplied summary.

2. Do NOT introduce outside legal knowledge.

3. Do NOT invent facts.

4. Do NOT invent legal provisions.

5. Do NOT invent arguments.

6. Do NOT invent court findings.

7. Do NOT invent conclusions.

8. Preserve important factual and legal information.

9. Preserve valid evidence references.

10. Keep legal provisions such as Article 368,
    Article 13(2), Article 31C, etc. recognizable.

11. Explain difficult legal terminology in simple language.

12. Do not change the legal meaning.

13. Clearly distinguish what the court said from
    simplified explanation.

14. Do not predict future court outcomes.

15. Do not provide personalized legal advice.

16. Do not tell the user what they should argue in court.

17. Preserve limitations contained in the source summary.

18. Do not claim that the explanation is exhaustive.

19. Make the output understandable to a non-lawyer.

20. Do not add facts that are not present in the supplied
    summary.

============================================================
SUPPLIED JUDGMENT SUMMARY
============================================================

The supplied summary is:

{summary_context}

============================================================
TASK
============================================================

Create a clear explanation of the supplied judgment summary
in {target_language}.

Translate the content naturally for a non-lawyer while
preserving the original legal meaning.

Every important factual or legal statement should retain its
corresponding VALID page reference from the supplied summary.

Do not use invalid citations.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "language": "{target_language}",
    "case_overview": "",
    "facts_of_case": [],
    "legal_issues": [],
    "important_arguments_or_content": [],
    "court_reasoning": [],
    "decision_or_conclusion": [],
    "important_legal_provisions": [],
    "simplified_explanation": "",
    "limitations": "",
    "disclaimer": "This AI-generated explanation is based only on the supplied judgment summary and is not legal advice. It should be verified against the original judgment."
}}

For each array, provide clear bullet-style text as strings.

The following fields should preserve relevant valid page
citations:

- case_overview
- facts_of_case
- legal_issues
- important_arguments_or_content
- court_reasoning
- decision_or_conclusion
- important_legal_provisions
- simplified_explanation

============================================================
FINAL CITATION CHECK
============================================================

Before returning the JSON:

1. Find every [Page X] citation.

2. Verify that every citation is between 1 and {total_pages}.

3. Verify that every citation exists in the valid citation
   list above.

4. Do NOT use [Page 2264].

5. Do NOT create any new citation.

6. Return ONLY valid JSON.

Return ONLY JSON.
"""

    return prompt


# ============================================================
# BUILD RETRY PROMPT
# ============================================================

def build_retry_prompt(
    summary: dict,
    language: str,
    previous_error: str,
    attempt_number: int,
    total_pages: int
) -> str:
    """
    Build a correction prompt after citation validation fails.
    """

    language_key = language.lower().strip()

    if language_key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    target_language = SUPPORTED_LANGUAGES[
        language_key
    ]

    summary_context = build_summary_context(
        summary
    )

    valid_source_citations = extract_valid_source_citations(
        summary,
        total_pages
    )

    citation_list = ", ".join(
        f"[Page {page}]"
        for page in valid_source_citations
    )

    if not citation_list:
        citation_list = "No valid page citations available."

    prompt = f"""
IMPORTANT: Your previous multilingual response failed
citation validation.

This is correction attempt {attempt_number}.

Previous validation error:

{previous_error}

============================================================
ORIGINAL JUDGMENT PAGE RANGE
============================================================

The original judgment contains exactly {total_pages} pages.

Therefore, every page citation MUST satisfy:

1 <= page number <= {total_pages}

============================================================
VALID CITATIONS
============================================================

The only valid authoritative citation tokens available
from the source summary are:

{citation_list}

============================================================
ABSOLUTE RULE
============================================================

DO NOT generate any citation outside the valid citation list.

In particular:

[Page 2264]

is INVALID and MUST NOT appear in your response.

Do not preserve it.

Do not reproduce it.

Do not mention it in the generated explanation.

Generate the COMPLETE corrected explanation.

============================================================
GENERAL RULES
============================================================

Use ONLY the supplied judgment summary.

Do not introduce outside information.

Do not invent facts.

Do not invent legal provisions.

Do not invent court findings.

Do not invent evidence.

Do not invent page numbers.

Preserve valid citations exactly.

Do not translate citations.

Do not alter citation numbers.

============================================================
TARGET LANGUAGE
============================================================

{target_language}

============================================================
SUPPLIED SUMMARY
============================================================

{summary_context}

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly:

{{
    "language": "{target_language}",
    "case_overview": "",
    "facts_of_case": [],
    "legal_issues": [],
    "important_arguments_or_content": [],
    "court_reasoning": [],
    "decision_or_conclusion": [],
    "important_legal_provisions": [],
    "simplified_explanation": "",
    "limitations": "",
    "disclaimer": "This AI-generated explanation is based only on the supplied judgment summary and is not legal advice. It should be verified against the original judgment."
}}

Every important factual or legal statement should preserve
its corresponding VALID page citation.

FINAL CHECK:

- No citation outside 1-{total_pages}.
- No [Page 2264].
- No invented citation.
- No changed citation.
- No translated citation.
- Return ONLY JSON.
"""

    return prompt


# ============================================================
# PARSE MODEL RESPONSE
# ============================================================

def parse_multilingual_response(
    response_text: str
) -> dict:
    """
    Parse Gemini JSON response.
    """

    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]

    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:

            json_candidate = cleaned[
                start:end + 1
            ]

            try:
                return json.loads(
                    json_candidate
                )

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Gemini returned an invalid multilingual JSON response."
        )


# ============================================================
# VALIDATE RESPONSE
# ============================================================

def validate_multilingual_response(
    result: dict,
    language: str
) -> dict:
    """
    Validate generated multilingual explanation.
    """

    required_fields = [
        "language",
        "case_overview",
        "facts_of_case",
        "legal_issues",
        "important_arguments_or_content",
        "court_reasoning",
        "decision_or_conclusion",
        "important_legal_provisions",
        "simplified_explanation",
        "limitations",
        "disclaimer"
    ]

    for field in required_fields:

        if field not in result:

            raise ValueError(
                f"Missing field in multilingual response: {field}"
            )

    language_key = language.lower().strip()

    if language_key not in SUPPORTED_LANGUAGES:

        raise ValueError(
            f"Unsupported language: {language}"
        )

    result["language"] = SUPPORTED_LANGUAGES[
        language_key
    ]

    array_fields = [
        "facts_of_case",
        "legal_issues",
        "important_arguments_or_content",
        "court_reasoning",
        "decision_or_conclusion",
        "important_legal_provisions"
    ]

    for field in array_fields:

        if not isinstance(
            result[field],
            list
        ):

            result[field] = [
                str(result[field])
            ]

    return result


# ============================================================
# GENERATE MULTILINGUAL EXPLANATION
# ============================================================

def generate_multilingual_explanation(
    summary: dict,
    language: str,
    total_pages: int
) -> dict:
    """
    Generate an evidence-grounded multilingual explanation.

    Automatically retries when citation validation fails.
    """

    # --------------------------------------------------------
    # Validate page count
    # --------------------------------------------------------

    if not isinstance(
        total_pages,
        int
    ) or total_pages <= 0:

        raise ValueError(
            "Invalid total_pages supplied for multilingual "
            "citation validation."
        )

    # --------------------------------------------------------
    # Validate language
    # --------------------------------------------------------

    language_key = language.lower().strip()

    if language_key not in SUPPORTED_LANGUAGES:

        raise ValueError(
            f"Unsupported language: {language}"
        )

    # --------------------------------------------------------
    # Build initial prompt
    # --------------------------------------------------------

    prompt = build_multilingual_prompt(
        summary=summary,
        language=language,
        total_pages=total_pages
    )

    last_error = None

    # --------------------------------------------------------
    # Generation + automatic retry
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_GENERATION_ATTEMPTS + 1
    ):

        try:

            # ------------------------------------------------
            # Call Gemini
            # ------------------------------------------------

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )

            response_text = interaction.output_text

            if not response_text:

                raise ValueError(
                    "Gemini returned an empty multilingual response."
                )

            # ------------------------------------------------
            # Parse response
            # ------------------------------------------------

            result = parse_multilingual_response(
                response_text
            )

            # ------------------------------------------------
            # Validate structure
            # ------------------------------------------------

            result = validate_multilingual_response(
                result,
                language
            )

            # ------------------------------------------------
            # Validate citations
            # ------------------------------------------------

            validate_citation_preservation(
                source_summary=summary,
                generated_result=result,
                total_pages=total_pages
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            return result

        
        except Exception as exc:
            last_error = str(exc)
            error_text = last_error.lower()

            # Do not retry when the API quota is exhausted.
            quota_indicators = (
                "resource_exhausted",
                "too_many_requests",
                "quota exceeded",
                "rate limit",
                "429",
            )

            if any(
                indicator in error_text
                for indicator in quota_indicators
            ):
                raise RuntimeError(
                    "Gemini API quota or rate limit reached. "
                    "Please wait for the quota to reset or use "
                    "an available model with remaining quota."
                ) from exc

            # Stop after the final generation attempt.
            if attempt >= MAX_GENERATION_ATTEMPTS:
                raise ValueError(
                    "Failed to generate a valid multilingual "
                    "explanation after "
                    f"{MAX_GENERATION_ATTEMPTS} attempts. "
                    f"Last error: {last_error}"
                ) from exc

            # Retry with a correction prompt.
            prompt = build_retry_prompt(
                summary=summary,
                language=language,
                previous_error=last_error,
                attempt_number=attempt + 1,
                total_pages=total_pages,
            )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    raise ValueError(
        "Multilingual generation failed unexpectedly. "
        f"Last error: {last_error}"
    )
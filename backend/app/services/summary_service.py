import json
import os
import time
import random

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
    api_key=API_KEY
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Your preferred model
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# Maximum number of judgment pages sent to Gemini.
#
# We intentionally keep this below 100 because very large
# prompts can increase the chance of transient 503 errors.
MAX_PAGES = 60


# Maximum characters sent to Gemini.
#
# This is another safety mechanism against extremely large
# judgments.
MAX_EVIDENCE_CHARS = 250000


# Number of attempts for transient errors.
MAX_RETRIES = 4


# Initial retry delay in seconds.
INITIAL_RETRY_DELAY = 2


# ============================================================
# FALLBACK MODEL PREFERENCES
# ============================================================

# These are preferences only.
#
# The code first checks which models are actually available
# for the current API key.
#
# If your configured model is unavailable or temporarily
# overloaded, another available model can be attempted.
#
# We intentionally include multiple generations because
# availability can vary by project/account.
MODEL_PREFERENCES = [
    MODEL_NAME,
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
]


# ============================================================
# DISCOVER AVAILABLE GEMINI MODELS
# ============================================================

def get_available_models() -> list:
    """
    Return models available to the current Gemini API key
    that support generateContent.

    This prevents us from blindly calling a model that the
    current API project cannot use.
    """

    available = []

    try:

        for model in client.models.list():

            name = getattr(
                model,
                "name",
                ""
            )

            supported_actions = getattr(
                model,
                "supported_actions",
                []
            )

            if not name:
                continue

            if "generateContent" not in supported_actions:
                continue

            # Convert:
            #
            # models/gemini-3.6-flash
            #
            # into:
            #
            # gemini-3.6-flash
            #
            clean_name = name.replace(
                "models/",
                ""
            )

            available.append(
                clean_name
            )

    except Exception as exc:

        print(
            "WARNING: Could not retrieve Gemini model list:"
        )

        print(exc)

    return available


# ============================================================
# BUILD MODEL CANDIDATE LIST
# ============================================================

def get_model_candidates() -> list:
    """
    Build an ordered list of usable model candidates.

    Preferred configured model comes first.
    """

    available_models = get_available_models()

    candidates = []

    # --------------------------------------------------------
    # First: configured / preferred models
    # --------------------------------------------------------

    for preferred in MODEL_PREFERENCES:

        if not preferred:
            continue

        preferred_clean = preferred.replace(
            "models/",
            ""
        )

        if preferred_clean in available_models:

            if preferred_clean not in candidates:

                candidates.append(
                    preferred_clean
                )

    # --------------------------------------------------------
    # Then: any other available Flash models
    # --------------------------------------------------------

    for available in available_models:

        lower_name = available.lower()

        if "flash" not in lower_name:
            continue

        if available not in candidates:

            candidates.append(
                available
            )

    # --------------------------------------------------------
    # If model listing failed
    # --------------------------------------------------------

    if not candidates:

        # At least try the configured model.
        candidates.append(
            MODEL_NAME.replace(
                "models/",
                ""
            )
        )

    return candidates


# ============================================================
# BUILD REPRESENTATIVE JUDGMENT CONTEXT
# ============================================================

def build_judgment_context(
    pages: list,
    max_pages: int = MAX_PAGES
) -> str:
    """
    Build a representative evidence context.

    Includes:

    - beginning pages
    - evenly distributed middle pages
    - ending pages

    Every page is explicitly labelled.

    The final context is also limited by
    MAX_EVIDENCE_CHARS.
    """

    if not pages:
        return ""

    total_pages = len(pages)

    # --------------------------------------------------------
    # Select pages
    # --------------------------------------------------------

    if total_pages <= max_pages:

        selected_pages = pages

    else:

        selected_indices = set()

        # ----------------------------------------------------
        # Beginning
        # ----------------------------------------------------

        beginning_count = min(
            15,
            max_pages // 4
        )

        for i in range(
            beginning_count
        ):

            selected_indices.add(i)

        # ----------------------------------------------------
        # Ending
        # ----------------------------------------------------

        ending_count = min(
            15,
            max_pages // 4
        )

        for i in range(
            total_pages - ending_count,
            total_pages
        ):

            selected_indices.add(i)

        # ----------------------------------------------------
        # Middle / distributed pages
        # ----------------------------------------------------

        remaining = (
            max_pages
            - len(selected_indices)
        )

        if remaining > 0:

            for i in range(
                remaining
            ):

                index = round(
                    i * (total_pages - 1)
                    / max(
                        remaining - 1,
                        1
                    )
                )

                selected_indices.add(
                    index
                )

        selected_indices = sorted(
            selected_indices
        )

        selected_indices = selected_indices[
            :max_pages
        ]

        selected_pages = [
            pages[i]
            for i in selected_indices
        ]

    # --------------------------------------------------------
    # Build evidence
    # --------------------------------------------------------

    context_parts = []

    current_length = 0

    for page in selected_pages:

        page_number = page.get(
            "page_number",
            "Unknown"
        )

        text = (
            page.get("cleaned_text")
            or page.get("text")
            or ""
        )

        if not text.strip():
            continue

        page_block = f"""
============================================================
JUDGMENT PAGE {page_number}
============================================================

{text}
"""

        block_length = len(
            page_block
        )

        # ----------------------------------------------------
        # Prevent gigantic requests
        # ----------------------------------------------------

        if (
            current_length
            + block_length
            > MAX_EVIDENCE_CHARS
        ):

            print(
                "Evidence character limit reached."
            )

            break

        context_parts.append(
            page_block
        )

        current_length += block_length

    return "\n".join(
        context_parts
    )


# ============================================================
# CLEAN MODEL JSON
# ============================================================

def clean_json_output(
    raw_output: str
) -> str:

    raw_output = (
        raw_output
        or ""
    ).strip()

    # --------------------------------------------------------
    # Remove ```json
    # --------------------------------------------------------

    if raw_output.startswith(
        "```json"
    ):

        raw_output = raw_output[
            len("```json"):
        ]

    # --------------------------------------------------------
    # Remove ```
    # --------------------------------------------------------

    elif raw_output.startswith(
        "```"
    ):

        raw_output = raw_output[
            len("```"):
        ]

    # --------------------------------------------------------
    # Remove closing ```
    # --------------------------------------------------------

    if raw_output.endswith(
        "```"
    ):

        raw_output = raw_output[
            :-len("```")
        ]

    return raw_output.strip()


# ============================================================
# GENERATE WITH RETRIES
# ============================================================

def generate_with_retry(
    prompt: str,
    model_name: str
):
    """
    Call Gemini with retry handling.

    Retries transient errors such as:

    503 UNAVAILABLE
    429 RESOURCE_EXHAUSTED
    500 INTERNAL
    502 BAD GATEWAY
    504 GATEWAY TIMEOUT

    Does NOT repeatedly retry 403 permission errors.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            print(
                f"Gemini attempt {attempt}/{MAX_RETRIES}"
                f" using {model_name}"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response

        except Exception as exc:

            last_error = exc

            error_text = str(
                exc
            )

            print(
                f"Gemini attempt {attempt} failed:"
            )

            print(
                error_text
            )

            # ------------------------------------------------
            # Determine whether the error is transient
            # ------------------------------------------------

            transient = any(
                code in error_text
                for code in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                    "502",
                    "BAD_GATEWAY",
                    "504",
                    "GATEWAY_TIMEOUT"
                ]
            )

            # ------------------------------------------------
            # Permission / authentication errors
            # ------------------------------------------------

            permission_error = any(
                code in error_text
                for code in [
                    "403",
                    "PERMISSION_DENIED",
                    "401",
                    "UNAUTHENTICATED"
                ]
            )

            if permission_error:

                print(
                    "Permission/authentication error."
                    " Trying another model if available."
                )

                break

            # ------------------------------------------------
            # Non-transient error
            # ------------------------------------------------

            if not transient:

                break

            # ------------------------------------------------
            # Last attempt
            # ------------------------------------------------

            if attempt >= MAX_RETRIES:

                break

            # ------------------------------------------------
            # Exponential backoff + jitter
            # ------------------------------------------------

            delay = (
                INITIAL_RETRY_DELAY
                * (2 ** (attempt - 1))
            )

            jitter = random.uniform(
                0,
                1
            )

            delay += jitter

            print(
                f"Retrying in {delay:.1f} seconds..."
            )

            time.sleep(
                delay
            )

    raise RuntimeError(
        f"Gemini model '{model_name}' failed "
        f"after {MAX_RETRIES} attempts.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# GENERATE JUDGMENT SUMMARY
# ============================================================

def generate_judgment_summary(
    pages: list,
    metadata: dict,
    max_pages: int = MAX_PAGES
) -> dict:
    """
    Generate a structured, evidence-grounded judgment summary.

    Uses Gemini generate_content.

    Includes:

    - model discovery
    - model fallback
    - retry handling
    - evidence sampling
    - JSON validation
    """

    # ========================================================
    # BUILD EVIDENCE
    # ========================================================

    evidence = build_judgment_context(
        pages,
        max_pages=max_pages
    )

    if not evidence:

        raise ValueError(
            "No judgment text available for summarization."
        )

    total_pages = len(
        pages
    )

    # ========================================================
    # CASE IDENTITY
    # ========================================================

    case_identity = json.dumps(
        {
            "case_number": metadata.get(
                "case_number"
            ),
            "petitioner": metadata.get(
                "petitioner"
            ),
            "respondent": metadata.get(
                "respondent"
            ),
            "judgment_date": metadata.get(
                "judgment_date"
            ),
            "bench": metadata.get(
                "bench"
            )
        },
        ensure_ascii=False,
        indent=2
    )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are an evidence-grounded legal judgment
summarization system.

Your task is to summarize the supplied Indian court
judgment using ONLY the judgment evidence provided below.

The original judgment contains {total_pages} pages.

============================================================
CASE IDENTITY
============================================================

The extracted case metadata is:

{case_identity}

Use this metadata only to identify the case.

Do not invent missing metadata.

============================================================
EVIDENCE RULES
============================================================

1. Use ONLY the supplied judgment evidence.

2. Do NOT use outside legal knowledge.

3. Do NOT invent facts.

4. Do NOT invent legal provisions.

5. Do NOT invent arguments.

6. Do NOT invent Court findings.

7. Do NOT infer the final decision from an isolated excerpt.

8. Preserve the distinction between:

- facts
- procedural background
- petitioner arguments
- respondent arguments
- counsel submissions
- judicial observations
- judicial reasoning
- final findings
- final decision

9. Never present a party's argument as a Court finding.

10. If the evidence says:

"It was argued"
"It was submitted"
"Counsel contended"
"The petitioner submitted"

identify the statement as an argument or submission.

11. Only describe something as a Court finding when
the supplied evidence clearly supports it.

============================================================
CASE-WIDE SUMMARY
============================================================

The judgment may contain many Articles, amendments,
constitutional provisions, arguments and legal questions.

Do NOT make one frequently mentioned topic the entire
case.

Identify the broader case context first.

Then explain the major legal questions and important
provisions actually supported by the evidence.

============================================================
LEGAL ISSUES
============================================================

Identify major legal questions actually discussed.

Do not convert every mentioned Article into a separate
legal issue.

============================================================
COURT REASONING
============================================================

The court_reasoning field must contain only reasoning
supported by the supplied judgment evidence.

Do NOT put party arguments into court_reasoning.

============================================================
DECISION
============================================================

The decision_or_conclusion field must contain the Court's
actual conclusion only when supported by the supplied
evidence.

Do NOT infer the final result from arguments.

If the available evidence does not establish the complete
final decision, explicitly say so.

============================================================
PAGE CITATIONS
============================================================

Every substantive factual or legal statement must contain
one or more page citations.

Use exactly:

[Page X]

or:

[Page X] [Page Y]

The page number MUST correspond to an actual page present
in the supplied evidence.

Never invent a page number.

============================================================
OUTPUT FORMAT
============================================================

Return exactly one JSON object with these fields:

{{
  "case_overview": "...",
  "facts_of_case": "...",
  "legal_issues": "...",
  "important_arguments_or_content": "...",
  "court_reasoning": "...",
  "decision_or_conclusion": "...",
  "important_legal_provisions": "...",
  "simplified_explanation": "...",
  "limitations": "...",
  "disclaimer": "..."
}}

============================================================
FIELD INSTRUCTIONS
============================================================

case_overview:

Give a short overview of the judgment as a whole.

Do not make one amendment or Article the entire case
unless the evidence clearly establishes that.

Every substantive statement requires a page citation.

------------------------------------------------------------

facts_of_case:

Describe relevant factual and procedural background
supported by the evidence.

Every substantive statement requires a page citation.

------------------------------------------------------------

legal_issues:

Identify the major constitutional/legal questions
actually discussed.

Every substantive statement requires a page citation.

------------------------------------------------------------

important_arguments_or_content:

Clearly attribute arguments to the relevant party,
counsel, or side when the evidence permits.

Do not convert arguments into Court findings.

Every substantive statement requires a page citation.

------------------------------------------------------------

court_reasoning:

Describe Court reasoning supported by the evidence.

Do not include unsupported legal conclusions.

Every substantive statement requires a page citation.

------------------------------------------------------------

decision_or_conclusion:

Describe the Court's conclusion or decision only when
supported by the evidence.

Every substantive statement requires a page citation.

------------------------------------------------------------

important_legal_provisions:

Identify important Articles, constitutional provisions,
and amendments actually discussed.

Every substantive statement requires a page citation.

------------------------------------------------------------

simplified_explanation:

Explain the judgment in simple language for a non-lawyer.

Preserve the legal meaning.

Every substantive statement requires a page citation.

------------------------------------------------------------

limitations:

Explain limitations caused by evidence sampling or
incomplete information.

If the evidence is incomplete, state that clearly.

Do not invent limitations.

------------------------------------------------------------

disclaimer:

Use exactly:

"This AI-generated summary is based only on the supplied judgment evidence and is not legal advice. It should be verified against the original judgment."

============================================================
JUDGMENT EVIDENCE
============================================================

{evidence}

============================================================
FINAL CHECK
============================================================

Before returning JSON, verify:

1. Correct case identity.
2. Overview represents the judgment as a whole.
3. No single amendment dominates without evidence.
4. Arguments are distinguished from Court findings.
5. No outside legal knowledge.
6. No unsupported conclusions.
7. Substantive statements have page citations.
8. Page numbers correspond to supplied pages.
9. Decision contains only supported conclusions.
10. Simplified explanation is faithful.

Return ONLY valid JSON.

Do NOT use Markdown.
Do NOT use code fences.
"""

    # ========================================================
    # FIND MODELS
    # ========================================================

    model_candidates = get_model_candidates()

    print(
        "============================================================"
    )

    print(
        "GEMINI SUMMARY MODEL CANDIDATES"
    )

    print(
        "============================================================"
    )

    for model in model_candidates:

        print(
            f"- {model}"
        )

    print(
        "============================================================"
    )

    # ========================================================
    # TRY MODELS
    # ========================================================

    errors = []

    for model_name in model_candidates:

        try:

            response = generate_with_retry(
                prompt=prompt,
                model_name=model_name
            )

            raw_output = (
                response.text
                or ""
            ).strip()

            if not raw_output:

                raise ValueError(
                    f"Model '{model_name}' returned "
                    "an empty response."
                )

            # ------------------------------------------------
            # Clean JSON
            # ------------------------------------------------

            raw_output = clean_json_output(
                raw_output
            )

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:

                summary = json.loads(
                    raw_output
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Model '{model_name}' returned "
                    f"invalid JSON: {exc}\n\n"
                    f"Raw response:\n"
                    f"{raw_output[:3000]}"
                )

            # ------------------------------------------------
            # Required fields
            # ------------------------------------------------

            required_fields = [
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

            missing_fields = [
                field
                for field in required_fields
                if field not in summary
            ]

            if missing_fields:

                raise ValueError(
                    "Summary response is missing fields: "
                    f"{missing_fields}"
                )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            print(
                "============================================================"
            )

            print(
                f"SUMMARY SUCCESSFULLY GENERATED "
                f"USING: {model_name}"
            )

            print(
                "============================================================"
            )

            return summary

        except Exception as exc:

            error_message = str(
                exc
            )

            errors.append(
                f"{model_name}: {error_message}"
            )

            print(
                "============================================================"
            )

            print(
                f"MODEL FAILED: {model_name}"
            )

            print(
                error_message
            )

            print(
                "Trying next available model..."
            )

            print(
                "============================================================"
            )

            continue

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    error_details = "\n\n".join(
        errors
    )

    raise RuntimeError(
        "Gemini summary generation failed with all "
        "available models.\n\n"
        f"{error_details}"
    )
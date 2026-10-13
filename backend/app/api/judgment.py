from backend.app.services.summary_service import generate_judgment_summary

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from pathlib import Path
import shutil
import uuid
import json
import re

from backend.app.services.pdf_service import extract_pdf_text

from backend.app.services.preprocessing_service import preprocess_judgment

from backend.app.services.rag_service import answer_question

from backend.app.services.document_index_service import build_document_index

from backend.app.services.legal_provision_service import (
    extract_legal_provisions_with_evidence,
    validate_legal_provisions,
    rank_legal_provisions
)

from backend.app.services.case_continuation_service import (
    generate_case_continuation_insights
)

from backend.app.services.multilingual_service import (
    generate_multilingual_explanation,
    SUPPORTED_LANGUAGES
)

from backend.app.services.provision_explanation_service import (
    generate_provision_explanation
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/judgment",
    tags=["Judgment"]
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uploads"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploads"
)

CASE_INSIGHTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_insights"
)

CASE_SUMMARIES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_summaries"
)

MULTILINGUAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multilingual_explanations"
)

PROVISION_EXPLANATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "provision_explanations"
)

DOCUMENT_INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CASE_INSIGHTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CASE_SUMMARIES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MULTILINGUAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROVISION_EXPLANATIONS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DOCUMENT_INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METADATA HELPERS
# ============================================================

def _first_non_empty(*values):
    """Return the first non-empty value."""

    for value in values:

        if value is None:
            continue

        if isinstance(value, str):

            if value.strip():
                return value.strip()

        elif value:
            return value

    return None


def _page_text(page):
    """Safely read text from a processed page."""

    if not isinstance(page, dict):
        return ""

    return str(
        page.get("text")
        or page.get("content")
        or page.get("page_text")
        or ""
    )


def _recover_case_metadata(processed_data):
    """
    Recover missing case metadata directly from the extracted
    judgment text.

    Supports common Indian Supreme Court judgment formats where
    the case title, date and bench are shown on the opening page
    without explicit labels such as 'DATED:' or 'CORAM:'.
    """

    if not isinstance(processed_data, dict):
        return processed_data

    metadata = processed_data.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}
        processed_data["metadata"] = metadata

    pages = processed_data.get("pages") or []

    if not pages:
        return processed_data

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def clean_text(value):
        if not value:
            return ""

        value = str(value)

        value = re.sub(
            r"\s+",
            " ",
            value
        )

        return value.strip(" :-,.")

    # --------------------------------------------------------
    # Collect opening pages
    # --------------------------------------------------------

    opening_pages = pages[:5]

    opening_text_parts = []

    for page in opening_pages:

        if not isinstance(page, dict):
            continue

        text = (
            page.get("text")
            or page.get("content")
            or page.get("page_text")
            or ""
        )

        if text:
            opening_text_parts.append(str(text))

    opening_text = "\n".join(
        opening_text_parts
    )

    # Normalize spaces but preserve lines.
    opening_text = re.sub(
        r"[ \t]+",
        " ",
        opening_text
    )

    # ========================================================
    # CASE NUMBER
    # ========================================================

    if not metadata.get("case_number"):

        case_patterns = [

            r"\b(Writ Petition\s*\(?(?:Civil|Criminal|Crl\.?)\)?"
            r"\s*No\.?\s*[A-Za-z0-9/(). -]+?\s+of\s+\d{4})",

            r"\b(Writ Petition\s+No\.?\s*"
            r"[A-Za-z0-9/(). -]+?\s+of\s+\d{4})",

            r"\b((?:Civil|Criminal|Special Leave|Review|Transfer|"
            r"Contempt)\s+Petition[^\n]{0,100}?"
            r"(?:No\.?\s*)?[A-Za-z0-9/(). -]+?\s+of\s+\d{4})"
        ]

        for pattern in case_patterns:

            match = re.search(
                pattern,
                opening_text,
                flags=re.IGNORECASE
            )

            if match:

                metadata["case_number"] = clean_text(
                    match.group(1)
                )

                break

    # ========================================================
    # JUDGMENT DATE
    # ========================================================

    if not metadata.get("judgment_date"):

        date_patterns = [

            # AUGUST 24, 2017
            r"\b("
            r"(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|"
            r"JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
            r"\s+\d{1,2},\s+\d{4}"
            r")",

            # August 24, 2017
            r"\b("
            r"(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r"\s+\d{1,2},\s+\d{4}"
            r")",

            # 24 August 2017
            r"\b("
            r"\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+\d{4}"
            r")",

            # 24/08/2017
            r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})"
        ]

        for pattern in date_patterns:

            match = re.search(
                pattern,
                opening_text,
                flags=re.IGNORECASE
            )

            if match:

                metadata["judgment_date"] = clean_text(
                    match.group(1)
                )

                break

    # ========================================================
    # PARTIES
    # ========================================================

    if (
        not metadata.get("petitioner")
        or not metadata.get("respondent")
    ):

        lines = []

        for line in opening_text.splitlines():

            line = clean_text(line)

            if line:
                lines.append(line)

        # ----------------------------------------------------
        # Find the "v." / "vs." / "versus" line
        # ----------------------------------------------------

        versus_index = None

        for index, line in enumerate(lines):

            normalized = (
                line
                .lower()
                .replace(" ", "")
                .replace(".", "")
            )

            if normalized in {
                "v",
                "vs",
                "versus"
            }:

                versus_index = index
                break

        if versus_index is not None:

            # ------------------------------------------------
            # Respondent
            # ------------------------------------------------

            if (
                versus_index + 1 < len(lines)
                and not metadata.get("respondent")
            ):

                respondent = lines[
                    versus_index + 1
                ]

                metadata["respondent"] = clean_text(
                    respondent
                )

            # ------------------------------------------------
            # Petitioner
            #
            # Sometimes the petitioner occupies multiple lines:
            #
            # JUSTICE K S PUTTASWAMY (RETD.),
            # AND ANR.
            # v.
            #
            # So combine the previous two lines.
            # ------------------------------------------------

            if not metadata.get("petitioner"):

                previous_lines = []

                start = max(
                    0,
                    versus_index - 2
                )

                for i in range(
                    start,
                    versus_index
                ):

                    previous_lines.append(
                        lines[i]
                    )

                if previous_lines:

                    petitioner = " ".join(
                        previous_lines
                    )

                    # Remove obvious report/citation lines.
                    if not re.match(
                        r"^\[?\d{4}\]?",
                        petitioner
                    ):

                        metadata["petitioner"] = (
                            clean_text(petitioner)
                        )

    # ========================================================
    # FIX FALSE PARTY EXTRACTION
    # ========================================================

    # If petitioner was incorrectly extracted as UNION OF INDIA
    # but the judgment clearly shows UNION OF INDIA after "v.",
    # correct it from the actual case title.

    if (
        metadata.get("petitioner")
        and metadata.get("petitioner").upper()
        in {
            "UNION OF INDIA",
            "UNION OF INDIA AND ORS.",
            "UNION OF INDIA AND ORS"
        }
    ):

        # Look specifically for the case-title structure.

        title_match = re.search(
            r"(?is)"
            r"([A-Z][A-Z0-9 .,&'()/-]{5,})"
            r"\s*\n\s*"
            r"(AND\s+(?:ANR|ORS)\.?)?"
            r"\s*\n\s*"
            r"v\.\s*\n\s*"
            r"(UNION OF INDIA(?:\s+AND\s+ORS\.?)?)",
            opening_text
        )

        if title_match:

            petitioner_parts = [
                title_match.group(1).strip()
            ]

            if title_match.group(2):
                petitioner_parts.append(
                    title_match.group(2).strip()
                )

            metadata["petitioner"] = clean_text(
                " ".join(petitioner_parts)
            )

            metadata["respondent"] = clean_text(
                title_match.group(3)
            )

    # ========================================================
    # CASE NAME
    # ========================================================

    if (
        metadata.get("petitioner")
        and metadata.get("respondent")
        and not metadata.get("case_name")
    ):

        metadata["case_name"] = (
            f"{metadata['petitioner']} "
            f"v. "
            f"{metadata['respondent']}"
        )

    # ========================================================
    # BENCH / JUDGES
    # ========================================================

    if not metadata.get("bench"):

        # Common Supreme Court format:
        #
        # AUGUST 24, 2017
        # [JAGDISH SINGH KHEHAR, CJI, J. CHELAMESWAR,
        # S. A. BOBDE, ... S. ABDUL NAZEER, JJ.]
        #
        # Extract the first bracketed block immediately
        # following the judgment date.

        date_match = re.search(
            r"\b("
            r"(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|"
            r"JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
            r"\s+\d{1,2},\s+\d{4}"
            r")",
            opening_text,
            flags=re.IGNORECASE
        )

        if date_match:

            text_after_date = opening_text[
                date_match.end():
            ]

            bench_match = re.search(
                r"\[\s*([^\]]{20,1000})\]",
                text_after_date
            )

            if bench_match:

                bench = clean_text(
                    bench_match.group(1)
                )

                metadata["bench"] = bench

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    for key in [
        "case_number",
        "judgment_date",
        "petitioner",
        "respondent",
        "bench",
        "case_name"
    ]:

        if metadata.get(key):

            metadata[key] = clean_text(
                metadata[key]
            )

    return processed_data


# ============================================================
# LLM ERROR HELPERS
# ============================================================

def _is_quota_error(error: Exception) -> bool:
    """
    Detect Gemini quota/rate-limit errors.
    """

    text = str(error).lower()

    quota_terms = [
        "429",
        "too_many_requests",
        "rate limit",
        "quota",
        "resource_exhausted",
        "exceeded",
        "quota exceeded"
    ]

    return any(
        term in text
        for term in quota_terms
    )


def _is_permission_error(error: Exception) -> bool:
    """
    Detect Gemini permission/access errors.
    """

    text = str(error).lower()

    permission_terms = [
        "403",
        "permission_denied",
        "permission denied",
        "project has been denied access",
        "access denied"
    ]

    return any(
        term in text
        for term in permission_terms
    )


def _is_model_not_found_error(error: Exception) -> bool:
    """
    Detect unavailable Gemini model errors.
    """

    text = str(error).lower()

    model_terms = [
        "404",
        "not_found",
        "model",
        "no longer available"
    ]

    return (
        any(
            term in text
            for term in model_terms
        )
        and (
            "model" in text
            or "not_found" in text
        )
    )


# ============================================================
# SAFE CONTINUATION FALLBACK
# ============================================================
def _continuation_quota_fallback(
    ranked_provisions: list,
    document_id: str
) -> dict:
    """
    Safe fallback when Gemini quota is exhausted.

    Uses only the provision names and metadata already
    produced by the extraction and ranking pipeline.
    Does not invent evidence or legal advice.
    """

    potential_legal_provisions = []
    issues_for_continuation = []
    related_provisions = []

    for item in ranked_provisions or []:
        if isinstance(item, str):
            provision_name = item.strip()
            provision_data = {"provision": provision_name}
        elif isinstance(item, dict):
            provision_data = item
            provision_name = str(
                item.get("provision", "")
            ).strip()
        else:
            continue

        if not provision_name:
            continue

        confidence = provision_data.get(
            "confidence", "unverified"
        )

        evidence_count = provision_data.get(
            "evidence_count", 0
        )

        page_count = provision_data.get(
            "page_count", 0
        )

        # Section 1: extracted provisions as research leads.
        potential_legal_provisions.append({
            "provision": provision_name,
            "confidence": confidence,
            "why_relevant": (
                "This provision was identified by the "
                "judgment's legal-provision extraction "
                "and ranking pipeline. Its relevance to "
                "any future proceedings requires review."
            ),
            "judgment_discussion": (
                f"The extraction pipeline reports "
                f"{evidence_count} evidence item(s) across "
                f"{page_count} page(s). The actual passages "
                "must be reviewed before drawing conclusions."
            ),
            "potential_continuation_issue": (
                "Research question: examine the judgment's "
                "discussion of this provision and determine "
                "whether it is relevant to any issue that "
                "remains in dispute. This is not a finding "
                "that an issue remains pending."
            ),
            "evidence": [],
            "limitation": (
                "The fallback does not have verified evidence "
                "snippets or page-level citations available "
                "from the ranked result."
            )
        })

        # Section 2: cautious research prompts, not legal claims.
        issues_for_continuation.append({
            "issue": (
                "Review the judgment's treatment of "
                + provision_name
            ),
            "description": (
                "Research task: locate and read the relevant "
                "passages, verify the applicable legal context, "
                "and establish whether any related question "
                "actually remains unresolved. The fallback "
                "cannot determine the status of a future hearing."
            ),
            "evidence": [],
            "confidence": "requires_manual_review",
            "limitation": (
                "This is a suggested research task based on "
                "an extracted provision, not an identified "
                "pending issue or a recommendation for a legal "
                "defence."
            )
        })

        # Section 3: display the ranked provisions.
        related_provisions.append({
            **provision_data,
            "relationship": (
                "Identified by the judgment's provision-ranking "
                "pipeline. A specific legal relationship with "
                "other provisions has not been independently "
                "established by this fallback."
            ),
            "evidence": []
        })

    return {
        "status": "llm_unavailable",
        "message": (
            "Gemini continuation analysis is unavailable "
            "because of a quota limit. The sections below "
            "contain extraction-based research leads only."
        ),
        "document_id": document_id,
        "potential_legal_provisions": potential_legal_provisions,
        "issues_for_continuation": issues_for_continuation,
        "related_provisions": related_provisions,
        "overall_limitation": (
            "These results are generated without the LLM. "
            "No evidence snippets, page citations, legal "
            "strategy, or predictions about future proceedings "
            "have been generated. Verify the original judgment "
            "and consult a qualified legal professional."
        ),
        "disclaimer": (
            "This output is not legal advice. Research leads "
            "must be checked against the original judgment "
            "and applicable law by a qualified legal professional."
        )
    }



# ============================================================
# UPLOAD JUDGMENT
# ============================================================

@router.post("/upload")
async def upload_judgment(
    file: UploadFile = File(...)
):
    """
    Upload and process an Indian court judgment PDF.

    Pipeline:

        PDF
         ↓
        Text extraction
         ↓
        Preprocessing
         ↓
        Metadata recovery
         ↓
        Document-specific chunks
         ↓
        Local embeddings
         ↓
        Document-specific FAISS index
    """

    # --------------------------------------------------------
    # 1. Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # --------------------------------------------------------
    # 2. Generate document ID
    # --------------------------------------------------------

    document_id = uuid.uuid4().hex

    safe_filename = Path(
        file.filename
    ).name

    upload_path = (
        UPLOAD_DIR
        / f"{document_id}_{safe_filename}"
    )

    # --------------------------------------------------------
    # 3. Save uploaded PDF
    # --------------------------------------------------------

    try:

        with upload_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save uploaded file: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # 4. Create processing directory
    # --------------------------------------------------------

    document_dir = (
        PROCESSED_DIR
        / document_id
    )

    document_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 5. Extract + preprocess
    # --------------------------------------------------------

    try:

        extracted_data = extract_pdf_text(
            str(upload_path)
        )

        processed_data = preprocess_judgment(
            extracted_data
        )

        processed_data = _recover_case_metadata(
            processed_data
        )

    except Exception as error:

        upload_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # 6. Save structured judgment
    # --------------------------------------------------------

    structured_path = (
        document_dir
        / "structured.json"
    )

    try:

        with structured_path.open(
            "w",
            encoding="utf-8"
        ) as output_file:

            json.dump(
                processed_data,
                output_file,
                indent=2,
                ensure_ascii=False
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save processed judgment: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # 7. Build document-specific RAG index
    # --------------------------------------------------------

    try:

        index_result = build_document_index(
            structured_file=structured_path,
            document_id=document_id
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to build document RAG index: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # 8. Return response
    # --------------------------------------------------------

    return {

        "message":
            "Judgment uploaded and processed successfully",

        "document_id":
            document_id,

        "filename":
            safe_filename,

        "total_pages":
            processed_data.get(
                "total_pages",
                0
            ),

        "metadata":
            processed_data.get(
                "metadata",
                {}
            ),

        "upload_path":
            str(upload_path),

        "structured_text_path":
            str(structured_path),

        "rag_index":
            index_result
    }


# ============================================================
# CASE CONTINUATION INSIGHTS
# ============================================================

@router.post(
    "/{document_id}/continuation-insights"
)
async def get_continuation_insights(
    document_id: str
):
    """
    Generate evidence-grounded case continuation insights.

    Local steps:
        Legal provision extraction
        ↓
        Validation
        ↓
        Ranking

    LLM step:
        Gemini/local LLM according to service configuration
    """

    # --------------------------------------------------------
    # STEP 1 — Locate judgment
    # --------------------------------------------------------

    document_dir = (
        PROCESSED_DIR
        / document_id
    )

    structured_path = (
        document_dir
        / "structured.json"
    )

    if not structured_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Processed judgment not found "
                f"for document_id: {document_id}"
            )
        )

    # --------------------------------------------------------
    # STEP 2 — Load judgment
    # --------------------------------------------------------

    try:

        with structured_path.open(
            "r",
            encoding="utf-8"
        ) as input_file:

            processed_data = json.load(
                input_file
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load processed judgment: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # STEP 3 — Cached result
    # --------------------------------------------------------

    insights_path = (
        CASE_INSIGHTS_DIR
        / f"{document_id}_continuation_insights.json"
    )

    if insights_path.exists():

        try:

            with insights_path.open(
                "r",
                encoding="utf-8"
            ) as input_file:

                cached_result = json.load(
                    input_file
                )

            return {

                "message":
                    "Existing case continuation insights returned",

                "document_id":
                    document_id,

                "cached":
                    True,

                "ranked_provisions":
                    cached_result.get(
                        "ranked_provisions",
                        []
                    ),

                "insights":
                    cached_result.get(
                        "insights",
                        {}
                    ),

                "output_path":
                    str(insights_path)
            }

        except Exception:

            pass

    # --------------------------------------------------------
    # STEP 4 — Local legal provision extraction
    # --------------------------------------------------------

    

    try:
        print(">>> STARTING PROVISION EXTRACTION")

        provisions = (
            extract_legal_provisions_with_evidence(
                processed_data.get(
                    "pages",
                    []
                )
            )
        )

        print(">>> PROVISION EXTRACTION COMPLETED")
        print(">>> PROVISIONS FOUND:", len(provisions))

        print(">>> STARTING PROVISION VALIDATION")

        validated_provisions = (
            validate_legal_provisions(
                provisions
            )
        )

        print(">>> PROVISION VALIDATION COMPLETED")

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Legal provision analysis failed: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # STEP 5 — Local ranking
    # --------------------------------------------------------
    
    try:
        print(">>> STARTING PROVISION RANKING")

        ranked_provisions = (
            rank_legal_provisions(
                validated_provisions,
                top_k=5
            )
        )
        print(">>> PROVISION RANKING COMPLETED")
        print(">>> RANKED PROVISIONS:", ranked_provisions)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Legal provision ranking failed: "
                f"{str(error)}"
            )
        )

    if not ranked_provisions:

        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant legal provisions "
                "were identified."
            )
        )

    # --------------------------------------------------------
    # STEP 6 — LLM continuation analysis
    # --------------------------------------------------------

    

    try:
        print(">>> STARTING CONTINUATION LLM")

        insights = (
            generate_case_continuation_insights(
                ranked_provisions,
                validated_provisions
            )
        )
        print(">>> CONTINUATION LLM COMPLETED")

    except Exception as error:

        print(
            "Continuation LLM error:",
            repr(error)
        )

        if _is_quota_error(error):

            insights = _continuation_quota_fallback(
                ranked_provisions,
                document_id
            )

        elif _is_permission_error(error):

            insights = {

                "status":
                    "llm_permission_error",

                "message": (
                    "The configured LLM project does not "
                    "currently have permission to access "
                    "the requested model/service."
                ),

                "document_id":
                    document_id,

                "related_provisions":
                    ranked_provisions,

                "disclaimer": (
                    "No model-generated legal analysis "
                    "was produced."
                )
            }

        elif _is_model_not_found_error(error):

            insights = {

                "status":
                    "llm_model_error",

                "message": (
                    "The configured LLM model is unavailable "
                    "for the current API project."
                ),

                "document_id":
                    document_id,

                "related_provisions":
                    ranked_provisions,

                "disclaimer": (
                    "No model-generated legal analysis "
                    "was produced."
                )
            }

        else:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Case continuation analysis failed: "
                    f"{str(error)}"
                )
            )

    # --------------------------------------------------------
    # STEP 7 — Prepare output
    # --------------------------------------------------------

    output = {

        "document_id":
            document_id,

        "document": {

            "filename":
                processed_data
                .get("metadata", {})
                .get("filename"),

            "total_pages":
                processed_data.get(
                    "total_pages"
                )
        },

        "ranked_provisions":
            ranked_provisions,

        "insights":
            insights
    }

    # --------------------------------------------------------
    # STEP 8 — Save result
    # --------------------------------------------------------

    try:

        with insights_path.open(
            "w",
            encoding="utf-8"
        ) as output_file:

            json.dump(
                output,
                output_file,
                indent=2,
                ensure_ascii=False
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save continuation insights: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # STEP 9 — Return result
    # --------------------------------------------------------

    return {

        "message":
            "Case continuation insights generated successfully",

        "document_id":
            document_id,

        "cached":
            False,

        "ranked_provisions":
            ranked_provisions,

        "insights":
            insights,

        "output_path":
            str(insights_path)
    }


# ============================================================
# JUDGMENT SUMMARY
# ============================================================

@router.post(
    "/{document_id}/summary"
)
def generate_summary(
    document_id: str
):
    """
    Generate an evidence-grounded judgment summary.
    """

    structured_file = (
        PROCESSED_DIR
        / document_id
        / "structured.json"
    )

    if not structured_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Processed judgment not found."
        )

    # --------------------------------------------------------
    # Load judgment
    # --------------------------------------------------------

    try:

        with structured_file.open(
            "r",
            encoding="utf-8"
        ) as input_file:

            structured_data = json.load(
                input_file
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load processed judgment: "
                f"{str(error)}"
            )
        )

    pages = structured_data.get(
        "pages",
        []
    )

    metadata = structured_data.get(
        "metadata",
        {}
    )

    if not pages:

        raise HTTPException(
            status_code=400,
            detail="No processed judgment pages found."
        )

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    try:

        summary = generate_judgment_summary(
            pages=pages,
            metadata=metadata,
            max_pages=80
        )

    except Exception as error:

        print(
            "Summary LLM error:",
            repr(error)
        )

        if _is_quota_error(error):

            raise HTTPException(
                status_code=429,
                detail={
                    "message": (
                        "Gemini quota/rate limit was reached "
                        "while generating the judgment summary."
                    ),
                    "document_id":
                        document_id,
                    "retryable":
                        True
                }
            )

        if _is_permission_error(error):

            raise HTTPException(
                status_code=403,
                detail={
                    "message": (
                        "The configured LLM project does not "
                        "have permission to access the requested "
                        "model/service."
                    ),
                    "document_id":
                        document_id
                }
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate judgment summary: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    output_file = (
        CASE_SUMMARIES_DIR
        / f"{document_id}_summary.json"
    )

    try:

        with output_file.open(
            "w",
            encoding="utf-8"
        ) as output_handle:

            json.dump(
                {
                    "document_id":
                        document_id,

                    "summary":
                        summary
                },
                output_handle,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save judgment summary: "
                f"{str(error)}"
            )
        )

    return {

        "message":
            "Judgment summary generated successfully",

        "document_id":
            document_id,

        "summary":
            summary,

        "output_path":
            str(output_file)
    }


# ============================================================
# MULTILINGUAL EXPLANATION
# ============================================================

@router.post(
    "/{document_id}/multilingual-explanation"
)
def generate_multilingual(
    document_id: str,
    language: str
):
    """
    Generate a multilingual explanation from
    the generated judgment summary.
    """

    summary_file = (
        CASE_SUMMARIES_DIR
        / f"{document_id}_summary.json"
    )

    if not summary_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Judgment summary not found. "
                "Generate the judgment summary first."
            )
        )

    language_key = (
        language.lower().strip()
    )

    if language_key not in SUPPORTED_LANGUAGES:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Unsupported language.",

                "supported_languages":
                    list(
                        SUPPORTED_LANGUAGES.values()
                    )
            }
        )

    # --------------------------------------------------------
    # Load summary
    # --------------------------------------------------------

    try:

        with summary_file.open(
            "r",
            encoding="utf-8"
        ) as input_file:

            summary_data = json.load(
                input_file
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load judgment summary: "
                f"{str(error)}"
            )
        )

    summary = summary_data.get(
        "summary"
    )

    if not summary:

        raise HTTPException(
            status_code=400,
            detail="Generated summary is empty."
        )

    # --------------------------------------------------------
    # Load page count
    # --------------------------------------------------------

    structured_file = (
        PROCESSED_DIR
        / document_id
        / "structured.json"
    )

    if not structured_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Processed judgment not found."
        )

    try:

        with structured_file.open(
            "r",
            encoding="utf-8"
        ) as input_file:

            structured_data = json.load(
                input_file
            )

        total_pages = structured_data.get(
            "total_pages",
            0
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load original judgment metadata: "
                f"{str(error)}"
            )
        )

    if not total_pages:

        raise HTTPException(
            status_code=500,
            detail="Original judgment page count is unavailable."
        )

    # --------------------------------------------------------
    # Generate multilingual explanation
    # --------------------------------------------------------

    try:

        explanation = (
            generate_multilingual_explanation(
                summary=summary,
                language=language_key,
                total_pages=total_pages
            )
        )

        print(
            "DEBUG LANGUAGE:",
            language_key
        )

        simplified_text = (
            explanation.get(
                "simplified_explanation",
                ""
            )
        )

        print(
            "DEBUG SIMPLIFIED:",
            repr(
                simplified_text[:500]
            )
        )

        print(
            "DEBUG TAMIL UNICODE COUNT:",
            sum(
                "\u0B80" <= char <= "\u0BFF"
                for char in simplified_text
            )
        )

    except Exception as error:

        print(
            "Multilingual LLM error:",
            repr(error)
        )

        if _is_quota_error(error):

            raise HTTPException(
                status_code=429,
                detail={
                    "message": (
                        "Gemini quota/rate limit was reached "
                        "while generating the multilingual explanation."
                    ),
                    "document_id":
                        document_id,
                    "retryable":
                        True
                }
            )

        if _is_permission_error(error):

            raise HTTPException(
                status_code=403,
                detail={
                    "message": (
                        "The configured LLM project does not "
                        "have permission to access the requested "
                        "model/service."
                    ),
                    "document_id":
                        document_id
                }
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate multilingual explanation: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # Save explanation
    # --------------------------------------------------------

    output_file = (
        MULTILINGUAL_DIR
        / f"{document_id}_{language_key}.json"
    )

    try:

        with output_file.open(
            "w",
            encoding="utf-8"
        ) as output_handle:

            json.dump(
                {
                    "document_id":
                        document_id,

                    "language":
                        SUPPORTED_LANGUAGES[
                            language_key
                        ],

                    "explanation":
                        explanation
                },
                output_handle,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save multilingual explanation: "
                f"{str(error)}"
            )
        )

    return {

        "message":
            "Multilingual explanation generated successfully",

        "document_id":
            document_id,

        "language":
            SUPPORTED_LANGUAGES[
                language_key
            ],

        "explanation":
            explanation,

        "output_path":
            str(output_file)
    }


# ============================================================
# LEGAL PROVISION EXPLANATION
# ============================================================

@router.post(
    "/{document_id}/provision-explanation"
)
def provision_explanation(
    document_id: str,
    provision: str
):
    """
    Generate an evidence-grounded explanation
    for a specific legal provision.
    """

    structured_file = (
        PROCESSED_DIR
        / document_id
        / "structured.json"
    )

    if not structured_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Processed judgment not found."
        )

    if not provision.strip():

        raise HTTPException(
            status_code=400,
            detail="Provision cannot be empty."
        )

    requested_provision = provision.strip()

    requested_lower = (
        requested_provision.lower()
    )

    # --------------------------------------------------------
    # Load judgment
    # --------------------------------------------------------

    try:

        with structured_file.open(
            "r",
            encoding="utf-8"
        ) as input_file:

            processed_data = json.load(
                input_file
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load processed judgment: "
                f"{str(error)}"
            )
        )

    pages = processed_data.get(
        "pages",
        []
    )

    if not pages:

        raise HTTPException(
            status_code=400,
            detail="No processed judgment pages found."
        )

    # --------------------------------------------------------
    # Extract + validate provisions locally
    # --------------------------------------------------------

    try:

        provisions = (
            extract_legal_provisions_with_evidence(
                pages
            )
        )

        validated_provisions = (
            validate_legal_provisions(
                provisions
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Legal provision extraction failed: "
                f"{str(error)}"
            )
        )

    if not isinstance(
        validated_provisions,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected legal provision data format. "
                "Expected a dictionary of provision categories."
            )
        )

    categories = [
        "articles",
        "sections",
        "constitutional_amendments"
    ]

    matching_provision = None
    matching_name = None

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    for category in categories:

        category_data = (
            validated_provisions.get(
                category,
                {}
            )
        )

        if not isinstance(
            category_data,
            dict
        ):
            continue

        for provision_name, provision_data in category_data.items():

            if (
                str(provision_name)
                .strip()
                .lower()
                == requested_lower
            ):

                matching_name = (
                    str(provision_name).strip()
                )

                matching_provision = (
                    provision_data
                )

                break

        if matching_provision is not None:
            break

    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    if matching_provision is None:

        for category in categories:

            category_data = (
                validated_provisions.get(
                    category,
                    {}
                )
            )

            if not isinstance(
                category_data,
                dict
            ):
                continue

            for provision_name, provision_data in category_data.items():

                extracted_lower = (
                    str(provision_name)
                    .strip()
                    .lower()
                )

                if (
                    requested_lower in extracted_lower
                    or extracted_lower in requested_lower
                ):

                    matching_name = (
                        str(provision_name).strip()
                    )

                    matching_provision = (
                        provision_data
                    )

                    break

            if matching_provision is not None:
                break

    if matching_provision is None:

        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    "Requested legal provision was not found "
                    "in the judgment."
                ),

                "provision":
                    requested_provision,

                "available_categories":
                    categories
            }
        )

    if not isinstance(
        matching_provision,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid data structure for the requested "
                "legal provision."
            )
        )

    # --------------------------------------------------------
    # Generate explanation using configured LLM
    # --------------------------------------------------------

    try:

        explanation = (
            generate_provision_explanation(
                provision=matching_name,
                provision_data=matching_provision
            )
        )

    except Exception as error:

        print(
            "Provision explanation LLM error:",
            repr(error)
        )

        if _is_quota_error(error):

            raise HTTPException(
                status_code=429,
                detail={
                    "message": (
                        "Gemini quota/rate limit was reached "
                        "while generating the provision explanation."
                    ),
                    "document_id":
                        document_id,
                    "provision":
                        matching_name,
                    "retryable":
                        True
                }
            )

        if _is_permission_error(error):

            raise HTTPException(
                status_code=403,
                detail={
                    "message": (
                        "The configured LLM project does not "
                        "have permission to access the requested "
                        "model/service."
                    ),
                    "document_id":
                        document_id,
                    "provision":
                        matching_name
                }
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate provision explanation: "
                f"{str(error)}"
            )
        )

    # --------------------------------------------------------
    # Save explanation
    # --------------------------------------------------------

    safe_provision_name = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        matching_name.lower()
    ).strip("_")

    output_file = (
        PROVISION_EXPLANATIONS_DIR
        / f"{document_id}_{safe_provision_name}.json"
    )

    output = {

        "document_id":
            document_id,

        "provision":
            matching_name,

        "explanation":
            explanation
    }

    try:

        with output_file.open(
            "w",
            encoding="utf-8"
        ) as output_handle:

            json.dump(
                output,
                output_handle,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save provision explanation: "
                f"{str(error)}"
            )
        )

    return {

        "message":
            "Legal provision explanation generated successfully",

        "document_id":
            document_id,

        "provision":
            matching_name,

        "explanation":
            explanation,

        "output_path":
            str(output_file)
    }


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

@router.post(
    "/{document_id}/ask"
)
def ask_question(
    document_id: str,
    question: str
):
    """
    Answer a question using evidence retrieved
    from the document-specific local RAG index.

    Hybrid architecture:

        User Question
              ↓
        Local Embedding
              ↓
        FAISS Retrieval
              ↓
        Relevant Judgment Chunks
              ↓
        Configured LLM
              ↓
        Grounded Answer
    """

    # --------------------------------------------------------
    # Verify judgment
    # --------------------------------------------------------

    structured_file = (
        PROCESSED_DIR
        / document_id
        / "structured.json"
    )

    if not structured_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Processed judgment not found."
        )

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # Verify document-specific FAISS index
    # --------------------------------------------------------

    document_index_dir = (
        DOCUMENT_INDEX_DIR
        / document_id
    )

    index_file = (
        document_index_dir
        / "legal_judgment.index"
    )

    metadata_file = (
        document_index_dir
        / "metadata.json"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Document-specific RAG index not found. "
                "Please upload the judgment again or "
                "rebuild its document index."
            )
        )

    if not metadata_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Document-specific RAG metadata not found."
            )
        )

    # --------------------------------------------------------
    # Hybrid RAG
    # --------------------------------------------------------

    try:

        result = answer_question(
            document_id=document_id,
            question=question.strip()
        )

    except Exception as error:

        print(
            "RAG LLM error:",
            repr(error)
        )

        if _is_quota_error(error):

            raise HTTPException(
                status_code=429,
                detail={
                    "message": (
                        "Gemini quota/rate limit was reached "
                        "while generating the answer."
                    ),
                    "document_id":
                        document_id,
                    "retryable":
                        True
                }
            )

        if _is_permission_error(error):

            raise HTTPException(
                status_code=403,
                detail={
                    "message": (
                        "The configured LLM project does not "
                        "have permission to access the requested "
                        "model/service."
                    ),
                    "document_id":
                        document_id
                }
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to answer question: "
                f"{str(error)}"
            )
        )

    return {

        "message":
            "Question answered successfully",

        "document_id":
            document_id,

        **result
    }
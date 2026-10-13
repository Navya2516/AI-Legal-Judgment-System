import re
from typing import Any, Dict, List


# ============================================================
# CANONICAL LEGAL TERMS
# ============================================================

AMENDMENT_PATTERNS = {
    "24th Amendment": [
        r"\b24th\s+amendment\b",
        r"\btwenty[-\s]?fourth\s+amendment\b",
        r"\bconstitution\s*\(\s*twenty[-\s]?fourth\s+amendment\s*\)\s*act\b",
    ],
    "25th Amendment": [
        r"\b25th\s+amendment\b",
        r"\btwenty[-\s]?fifth\s+amendment\b",
        r"\bconstitution\s*\(\s*twenty[-\s]?fifth\s+amendment\s*\)\s*act\b",
    ],
    "26th Amendment": [
        r"\b26th\s+amendment\b",
        r"\btwenty[-\s]?sixth\s+amendment\b",
        r"\bconstitution\s*\(\s*twenty[-\s]?sixth\s+amendment\s*\)\s*act\b",
    ],
    "29th Amendment": [
        r"\b29th\s+amendment\b",
        r"\btwenty[-\s]?ninth\s+amendment\b",
        r"\bconstitution\s*\(\s*twenty[-\s]?ninth\s+amendment\s*\)\s*act\b",
    ],
}


ARTICLE_PATTERN = re.compile(
    r"\bArticle\s+"
    r"(\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))?)",
    re.IGNORECASE
)


SECTION_PATTERN = re.compile(
    r"\bSection\s+"
    r"(\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))?)",
    re.IGNORECASE
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the actual wording.
    """

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ============================================================
# AMENDMENT EXTRACTION
# ============================================================

def extract_amendments(
    evidence_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract constitutional amendments directly mentioned
    in the retrieved evidence.

    This is deterministic and does NOT use an LLM.
    """

    extracted = {}

    for chunk in evidence_chunks:

        text = normalize_text(
            chunk.get("text", "")
        )

        page_number = chunk.get(
            "page_number"
        )

        for canonical, patterns in AMENDMENT_PATTERNS.items():

            matched = any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

            if not matched:
                continue

            if canonical not in extracted:

                extracted[canonical] = {
                    "name": canonical,
                    "mentioned": True,
                    "evidence_pages": [],
                    "evidence_chunks": [],
                }

            if (
                page_number is not None
                and page_number not in extracted[
                    canonical
                ]["evidence_pages"]
            ):

                extracted[
                    canonical
                ]["evidence_pages"].append(
                    page_number
                )

            chunk_id = chunk.get(
                "chunk_id"
            )

            if (
                chunk_id
                and chunk_id not in extracted[
                    canonical
                ]["evidence_chunks"]
            ):

                extracted[
                    canonical
                ]["evidence_chunks"].append(
                    chunk_id
                )

    return list(
        extracted.values()
    )


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_articles(
    evidence_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract Article references directly from evidence.

    Example:
        Article 368
        Article 13
        Article 31C
    """

    extracted = {}

    for chunk in evidence_chunks:

        text = normalize_text(
            chunk.get("text", "")
        )

        page_number = chunk.get(
            "page_number"
        )

        matches = ARTICLE_PATTERN.findall(
            text
        )

        for match in matches:

            article = f"Article {match}"

            key = article.lower()

            if key not in extracted:

                extracted[key] = {
                    "name": article,
                    "evidence_pages": [],
                    "evidence_chunks": [],
                }

            if (
                page_number is not None
                and page_number not in extracted[
                    key
                ]["evidence_pages"]
            ):

                extracted[
                    key
                ]["evidence_pages"].append(
                    page_number
                )

            chunk_id = chunk.get(
                "chunk_id"
            )

            if (
                chunk_id
                and chunk_id not in extracted[
                    key
                ]["evidence_chunks"]
            ):

                extracted[
                    key
                ]["evidence_chunks"].append(
                    chunk_id
                )

    return list(
        extracted.values()
    )


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_sections(
    evidence_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract statutory Section references directly from
    retrieved evidence.
    """

    extracted = {}

    for chunk in evidence_chunks:

        text = normalize_text(
            chunk.get("text", "")
        )

        page_number = chunk.get(
            "page_number"
        )

        matches = SECTION_PATTERN.findall(
            text
        )

        for match in matches:

            section = f"Section {match}"

            key = section.lower()

            if key not in extracted:

                extracted[key] = {
                    "name": section,
                    "evidence_pages": [],
                    "evidence_chunks": [],
                }

            if (
                page_number is not None
                and page_number not in extracted[
                    key
                ]["evidence_pages"]
            ):

                extracted[
                    key
                ]["evidence_pages"].append(
                    page_number
                )

            chunk_id = chunk.get(
                "chunk_id"
            )

            if (
                chunk_id
                and chunk_id not in extracted[
                    key
                ]["evidence_chunks"]
            ):

                extracted[
                    key
                ]["evidence_chunks"].append(
                    chunk_id
                )

    return list(
        extracted.values()
    )


# ============================================================
# BUILD STRUCTURED EVIDENCE
# ============================================================

def build_evidence_facts(
    evidence_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert retrieved evidence into deterministic,
    structured facts.

    The function does not infer legal meaning.
    It only records information explicitly present
    in the retrieved chunks.
    """

    return {
        "amendments": extract_amendments(
            evidence_chunks
        ),

        "articles": extract_articles(
            evidence_chunks
        ),

        "sections": extract_sections(
            evidence_chunks
        ),

        "evidence_count": len(
            evidence_chunks
        ),

        "pages": sorted(
            list({
                chunk.get("page_number")
                for chunk in evidence_chunks
                if chunk.get("page_number")
                is not None
            })
        ),
    }


# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def validate_answer_against_evidence(
    answer: str,
    evidence_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform lightweight deterministic validation.

    This does NOT attempt to understand the entire legal
    meaning of the answer.

    It checks whether important legal references appearing
    in the answer are present in the retrieved evidence.
    """

    evidence_text = " ".join(
        normalize_text(
            chunk.get("text", "")
        )
        for chunk in evidence_chunks
    ).lower()

    answer_text = normalize_text(
        answer
    )

    unsupported_articles = []
    unsupported_amendments = []
    unsupported_sections = []

    # --------------------------------------------------------
    # Check Articles
    # --------------------------------------------------------

    answer_articles = ARTICLE_PATTERN.findall(
        answer_text
    )

    for article_number in set(
        answer_articles
    ):

        article_reference = (
            f"article {article_number}"
        ).lower()

        if article_reference not in evidence_text:

            unsupported_articles.append(
                f"Article {article_number}"
            )

    # --------------------------------------------------------
    # Check Sections
    # --------------------------------------------------------

    answer_sections = SECTION_PATTERN.findall(
        answer_text
    )

    for section_number in set(
        answer_sections
    ):

        section_reference = (
            f"section {section_number}"
        ).lower()

        if section_reference not in evidence_text:

            unsupported_sections.append(
                f"Section {section_number}"
            )

    # --------------------------------------------------------
    # Check Amendments
    # --------------------------------------------------------

    for canonical, patterns in AMENDMENT_PATTERNS.items():

        if not re.search(
            re.escape(
                canonical.split()[0]
            ),
            answer_text,
            re.IGNORECASE
        ):
            continue

        answer_mentions_amendment = any(
            re.search(
                pattern,
                answer_text,
                re.IGNORECASE
            )
            for pattern in patterns
        )

        if not answer_mentions_amendment:
            continue

        evidence_mentions_amendment = any(
            re.search(
                pattern,
                evidence_text,
                re.IGNORECASE
            )
            for pattern in patterns
        )

        if not evidence_mentions_amendment:

            unsupported_amendments.append(
                canonical
            )

    total_unsupported = (
        len(unsupported_articles)
        + len(unsupported_sections)
        + len(unsupported_amendments)
    )

    return {
        "is_supported": (
            total_unsupported == 0
        ),

        "unsupported_articles": (
            unsupported_articles
        ),

        "unsupported_sections": (
            unsupported_sections
        ),

        "unsupported_amendments": (
            unsupported_amendments
        ),

        "unsupported_count": (
            total_unsupported
        ),
    }
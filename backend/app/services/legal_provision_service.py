import re
import math


# ============================================================
# REGEX PATTERNS
# ============================================================


ARTICLE_PATTERN = re.compile(
    r"\bArticles?\s+"
    r"("
    r"\d+(?:\s*[-–]\s*[A-Za-z]|[A-Za-z])?"
    r"(?:"
    r"\s*(?:and|to|-)\s*"
    r"\d+(?:\s*[-–]\s*[A-Za-z]|[A-Za-z])?"
    r")*"
    r")"
    r"\b",
    re.IGNORECASE
)


SECTION_PATTERN = re.compile(
    r"\bSections?\s+"
    r"(\d+(?:\s*(?:and|to|-)\s*\d+)*)"
    r"\b",
    re.IGNORECASE
)


AMENDMENT_PATTERN = re.compile(
    r"\b"
    r"(?:(?:Constitution\s*\()?"
    r"(?:Twenty[- ]?First|Twenty[- ]?Second|"
    r"Twenty[- ]?Third|Twenty[- ]?Fourth|"
    r"Twenty[- ]?Fifth|Twenty[- ]?Sixth|"
    r"Twenty[- ]?Seventh|Twenty[- ]?Eighth|"
    r"Twenty[- ]?Ninth|"
    r"\d+(?:st|nd|rd|th))"
    r"\s+Amendment"
    r"(?:\s+Act)?"
    r"(?:\))?)",
    re.IGNORECASE
)


# ============================================================
# WORD → NUMBER MAPPING
# ============================================================

WORD_TO_NUMBER = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,

    "twenty-first": 21,

    "twenty second": 22,
    "twenty-second": 22,

    "twenty third": 23,
    "twenty-third": 23,

    "twenty fourth": 24,
    "twenty-fourth": 24,

    "twenty fifth": 25,
    "twenty-fifth": 25,

    "twenty sixth": 26,
    "twenty-sixth": 26,

    "twenty seventh": 27,
    "twenty-seventh": 27,

    "twenty eighth": 28,
    "twenty-eighth": 28,

    "twenty ninth": 29,
    "twenty-ninth": 29,
}


# ============================================================
# AMENDMENT ACT YEARS
# ============================================================

AMENDMENT_ACT_YEARS = {
    1: 1951,
    4: 1955,
    10: 1961,
    14: 1962,
    15: 1963,
    16: 1963,
    17: 1964,
    18: 1966,
    19: 1966,
    21: 1967,
    24: 1971,
    25: 1971,
    26: 1971,
    29: 1972,
}


# ============================================================
# STANDARD AMENDMENT NAMES
# ============================================================

ORDINAL_NAMES = {
    1: "First",
    4: "Fourth",
    10: "Tenth",
    14: "Fourteenth",
    15: "Fifteenth",
    16: "Sixteenth",
    17: "Seventeenth",
    18: "Eighteenth",
    19: "Nineteenth",
    21: "Twenty-first",
    24: "Twenty-fourth",
    25: "Twenty-fifth",
    26: "Twenty-sixth",
    29: "Twenty-ninth",
}


# ============================================================
# AMENDMENT NORMALIZATION
# ============================================================

def normalize_amendment(match_text: str):
    """
    Convert different amendment name formats into
    one standard representation.

    Example:
        24th Amendment
        Twenty-fourth Amendment
        Constitution (Twenty-fourth Amendment) Act

    become:

        Constitution (Twenty-fourth Amendment) Act, 1971
    """

    text = " ".join(match_text.split())

    # Remove "Constitution (" prefix
    text = re.sub(
        r"^Constitution\s*\(\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove closing parenthesis
    text = text.replace(")", "")

    # Remove trailing "Act"
    text = re.sub(
        r"\s+Act$",
        "",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Numeric ordinal form
    # --------------------------------------------------------

    ordinal_match = re.search(
        r"\b(\d+)(?:st|nd|rd|th)\s+Amendment\b",
        text,
        re.IGNORECASE
    )

    if ordinal_match:

        number = int(
            ordinal_match.group(1)
        )

    else:

        # ----------------------------------------------------
        # Written ordinal form
        # ----------------------------------------------------

        word_match = re.search(
            r"\b("
            r"Twenty[- ]?First|"
            r"Twenty[- ]?Second|"
            r"Twenty[- ]?Third|"
            r"Twenty[- ]?Fourth|"
            r"Twenty[- ]?Fifth|"
            r"Twenty[- ]?Sixth|"
            r"Twenty[- ]?Seventh|"
            r"Twenty[- ]?Eighth|"
            r"Twenty[- ]?Ninth"
            r")\s+Amendment\b",
            text,
            re.IGNORECASE
        )

        if not word_match:
            return None

        word = (
            word_match.group(1)
            .lower()
            .replace(" ", "-")
        )

        number = WORD_TO_NUMBER.get(
            word
        )

        if number is None:
            return None

    # --------------------------------------------------------
    # Determine year
    # --------------------------------------------------------

    year = AMENDMENT_ACT_YEARS.get(
        number
    )

    name = ORDINAL_NAMES.get(
        number,
        f"{number}th"
    )

    # --------------------------------------------------------
    # Return normalized name
    # --------------------------------------------------------

    if year:

        return (
            f"Constitution "
            f"({name} Amendment) Act, "
            f"{year}"
        )

    return (
        f"Constitution "
        f"({name} Amendment) Act"
    )


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_articles(text: str) -> list:
    """
    Extract Constitution Article references from text.

    Example:
        Article 14
        Articles 14 and 19

    returns:
        ["Article 14", "Article 19"]
    """

    articles = set()

    matches = ARTICLE_PATTERN.findall(
        text
    )

    for match in matches:

        numbers = re.findall(
            r"\d+",
            match
        )

        for number in numbers:

            number = int(number)

            # Constitution currently has articles
            # within this broad range.
            if 1 <= number <= 395:

                articles.add(
                    f"Article {number}"
                )

    return sorted(
        articles,
        key=lambda x: int(
            re.search(
                r"\d+",
                x
            ).group()
        )
    )


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_sections(text: str) -> list:
    """
    Extract Section references from text.

    Example:
        Section 2
        Sections 2 and 3
    """

    sections = set()

    matches = SECTION_PATTERN.findall(
        text
    )

    for match in matches:

        numbers = re.findall(
            r"\d+",
            match
        )

        for number in numbers:

            number = int(number)

            if 1 <= number <= 1000:

                sections.add(
                    f"Section {number}"
                )

    return sorted(
        sections,
        key=lambda x: int(
            re.search(
                r"\d+",
                x
            ).group()
        )
    )


# ============================================================
# AMENDMENT EXTRACTION
# ============================================================

def extract_amendments(text: str) -> list:
    """
    Extract and normalize constitutional amendment
    references.
    """

    amendments = set()

    matches = AMENDMENT_PATTERN.findall(
        text
    )

    for match in matches:

        normalized = normalize_amendment(
            match
        )

        if normalized:

            amendments.add(
                normalized
            )

    return sorted(
        amendments
    )


# ============================================================
# BASIC LEGAL PROVISION EXTRACTION
# ============================================================

def extract_legal_provisions(
    pages: list
) -> dict:
    """
    Extract Articles, Sections and Constitutional
    Amendments from all pages.

    Returns page-level occurrence information.
    """

    article_pages = {}
    section_pages = {}
    amendment_pages = {}

    for page in pages:

        page_number = page[
            "page_number"
        ]

        text = page[
            "text"
        ]

        # ----------------------------------------------------
        # Articles
        # ----------------------------------------------------

        articles = extract_articles(
            text
        )

        for article in articles:

            article_pages.setdefault(
                article,
                []
            ).append(
                page_number
            )

        # ----------------------------------------------------
        # Sections
        # ----------------------------------------------------

        sections = extract_sections(
            text
        )

        for section in sections:

            section_pages.setdefault(
                section,
                []
            ).append(
                page_number
            )

        # ----------------------------------------------------
        # Amendments
        # ----------------------------------------------------

        amendments = extract_amendments(
            text
        )

        for amendment in amendments:

            amendment_pages.setdefault(
                amendment,
                []
            ).append(
                page_number
            )

    return {
        "articles": article_pages,

        "sections": section_pages,

        "constitutional_amendments":
            amendment_pages
    }


# ============================================================
# EXTRACTION WITH EVIDENCE
# ============================================================

def extract_legal_provisions_with_evidence(
    pages: list
) -> dict:
    """
    Extract legal provisions together with
    page-level evidence.

    Evidence contains approximately 200 characters
    before and after each provision reference.
    """

    articles = {}
    sections = {}
    amendments = {}

    for page in pages:

        page_number = page[
            "page_number"
        ]

        text = page[
            "text"
        ]

        # ====================================================
        # ARTICLES
        # ====================================================

        article_matches = ARTICLE_PATTERN.finditer(
            text
        )

        for match in article_matches:

            article_expression = match.group(1)

            # ----------------------------------------------------
            # Extract Article references carefully.
            #
            # Valid examples:
            #   13, 19, 31, 31A, 31B, 31C, 31-A, 368
            #
            # Reject OCR-style malformed references such as:
            #   14T, 368I, etc.
            # ----------------------------------------------------

            article_parts = re.findall(
                r"\d+\s*(?:[-–]\s*[A-Za-z]|[A-Za-z])?",
                article_expression
            )

            for article_part in article_parts:

                article_part = article_part.strip()

                # Normalize hyphen spacing
                article_part = re.sub(
                    r"\s*[-–]\s*",
                    "-",
                    article_part
                )

                # Normalize "31 A" -> "31A"
                article_part = re.sub(
                    r"(\d+)\s+([A-Za-z])$",
                    r"\1\2",
                    article_part
                )

                # ------------------------------------------------
                # Separate numeric part and suffix
                # ------------------------------------------------

                number_match = re.fullmatch(
                    r"(\d+)([A-Za-z]?)",
                    article_part
                )

                if not number_match:
                    continue

                number = int(
                    number_match.group(1)
                )

                suffix = (
                    number_match.group(2).upper()
                )

                # ------------------------------------------------
                # Valid Constitution Article range
                # ------------------------------------------------

                if not (
                    1 <= number <= 395
                ):
                    continue

                # ------------------------------------------------
                # Only accept known constitutional
                # article suffixes.
                # ------------------------------------------------

                VALID_ARTICLE_SUFFIXES = {
                    31: {"A", "B", "C"},
                    32: {"A"},
                    43: {"A"},
                    51: {"A"},
                    239: {"A", "AA", "AB"},
                    243: {
                        "A", "B", "C", "D", "E", "F", "G",
                        "H", "I", "J", "K", "L", "M", "N",
                        "O", "P", "Q", "R", "S", "T", "U",
                        "V", "W", "X", "Y", "Z"
                    },
                    300: {"A"},
                    323: {"A"},
                    324: {"A"},
                    342: {"A"},
                    371: {
                        "A", "B", "C", "D", "E", "F", "G",
                        "H", "I", "J"
                    }
                }

                if suffix:

                    allowed_suffixes = (
                        VALID_ARTICLE_SUFFIXES.get(
                            number,
                            set()
                        )
                    )

                    if suffix not in allowed_suffixes:
                        continue

                provision = (
                    f"Article {number}{suffix}"
                )

                start = max(
                    0,
                    match.start() - 200
                )

                end = min(
                    len(text),
                    match.end() + 200
                )

                evidence_text = (
                    text[
                        start:end
                    ].strip()
                )

                if provision not in articles:

                    articles[
                        provision
                    ] = {
                        "pages": [],
                        "evidence": []
                    }

                if page_number not in (
                    articles[
                        provision
                    ]["pages"]
                ):

                    articles[
                        provision
                    ]["pages"].append(
                        page_number
                    )

                articles[
                    provision
                ]["evidence"].append(
                    {
                        "page": page_number,
                        "text": evidence_text
                    }
                )

             


        # ====================================================
        # SECTIONS
        # ====================================================

        section_matches = SECTION_PATTERN.finditer(
            text
        )

        for match in section_matches:

            numbers = re.findall(
                r"\d+",
                match.group(1)
            )

            for number in numbers:

                number = int(number)

                if not (
                    1 <= number <= 1000
                ):
                    continue

                provision = (
                    f"Section {number}"
                )

                start = max(
                    0,
                    match.start() - 200
                )

                end = min(
                    len(text),
                    match.end() + 200
                )

                evidence_text = (
                    text[
                        start:end
                    ].strip()
                )

                if provision not in sections:

                    sections[
                        provision
                    ] = {
                        "pages": [],
                        "evidence": []
                    }

                if page_number not in (
                    sections[
                        provision
                    ]["pages"]
                ):

                    sections[
                        provision
                    ]["pages"].append(
                        page_number
                    )

                sections[
                    provision
                ]["evidence"].append(
                    {
                        "page": page_number,
                        "text": evidence_text
                    }
                )

        # ====================================================
        # CONSTITUTIONAL AMENDMENTS
        # ====================================================

        amendment_matches = AMENDMENT_PATTERN.finditer(
            text
        )

        for match in amendment_matches:

            normalized = normalize_amendment(
                match.group(0)
            )

            if not normalized:
                continue

            start = max(
                0,
                match.start() - 200
            )

            end = min(
                len(text),
                match.end() + 200
            )

            evidence_text = (
                text[
                    start:end
                ].strip()
            )

            if normalized not in amendments:

                amendments[
                    normalized
                ] = {
                    "pages": [],
                    "evidence": []
                }

            if page_number not in (
                amendments[
                    normalized
                ]["pages"]
            ):

                amendments[
                    normalized
                ]["pages"].append(
                    page_number
                )

            amendments[
                normalized
            ]["evidence"].append(
                {
                    "page": page_number,
                    "text": evidence_text
                }
            )

    return {
        "articles": articles,

        "sections": sections,

        "constitutional_amendments":
            amendments
    }


# ============================================================
# EVIDENCE CONFIDENCE
# ============================================================

def calculate_evidence_confidence(
    evidence_text: str,
    provision: str
) -> str:
    """
    Assign evidence confidence using simple
    rule-based legal context signals.

    High:
        Strong legal context + provision reference

    Medium:
        Some legal context

    Low:
        Weak contextual evidence
    """

    text = evidence_text.lower()

    provision_lower = (
        provision.lower()
    )

    score = 0

    # --------------------------------------------------------
    # Exact provision match
    # --------------------------------------------------------

    if provision_lower in text:

        score += 2

    # --------------------------------------------------------
    # Legal keywords
    # --------------------------------------------------------

    legal_keywords = [
        "constitution",
        "constitutional",
        "amendment",
        "validity",
        "challenge",
        "power",
        "rights",
        "court",
        "judgment",
        "provision",
        "law",
        "act",
        "petition",
        "article",
        "section"
    ]

    keyword_count = sum(
        1
        for keyword in legal_keywords
        if keyword in text
    )

    if keyword_count >= 3:

        score += 2

    elif keyword_count >= 1:

        score += 1

    # --------------------------------------------------------
    # Strong legal phrases
    # --------------------------------------------------------

    strong_phrases = [
        "validity of",
        "constitutional validity",
        "challenge to",
        "power of parliament",
        "power of parliament to",
        "contrary to",
        "violates",
        "violation of",
        "impugned",
        "held that",
        "court held",
        "court observed",
        "question before"
    ]

    if any(
        phrase in text
        for phrase in strong_phrases
    ):

        score += 2

    # --------------------------------------------------------
    # Final confidence
    # --------------------------------------------------------

    if score >= 5:

        return "high"

    if score >= 3:

        return "medium"

    return "low"


# ============================================================
# VALIDATE LEGAL PROVISIONS
# ============================================================

def validate_legal_provisions(
    provisions: dict
) -> dict:
    """
    Add confidence information to every
    evidence record.
    """

    validated = {
        "articles": {},
        "sections": {},
        "constitutional_amendments": {}
    }

    # ========================================================
    # ARTICLES
    # ========================================================

    for article, data in (
        provisions[
            "articles"
        ].items()
    ):

        evidence_records = []

        for evidence in data[
            "evidence"
        ]:

            confidence = (
                calculate_evidence_confidence(
                    evidence["text"],
                    article
                )
            )

            evidence_records.append(
                {
                    "page": evidence["page"],
                    "text": evidence["text"],
                    "confidence": confidence
                }
            )

        validated[
            "articles"
        ][article] = {
            "pages": data["pages"],
            "evidence": evidence_records
        }

    # ========================================================
    # SECTIONS
    # ========================================================

    for section, data in (
        provisions[
            "sections"
        ].items()
    ):

        evidence_records = []

        for evidence in data[
            "evidence"
        ]:

            confidence = (
                calculate_evidence_confidence(
                    evidence["text"],
                    section
                )
            )

            evidence_records.append(
                {
                    "page": evidence["page"],
                    "text": evidence["text"],
                    "confidence": confidence
                }
            )

        validated[
            "sections"
        ][section] = {
            "pages": data["pages"],
            "evidence": evidence_records
        }

    # ========================================================
    # CONSTITUTIONAL AMENDMENTS
    # ========================================================

    for amendment, data in (
        provisions[
            "constitutional_amendments"
        ].items()
    ):

        evidence_records = []

        for evidence in data[
            "evidence"
        ]:

            confidence = (
                calculate_evidence_confidence(
                    evidence["text"],
                    amendment
                )
            )

            evidence_records.append(
                {
                    "page": evidence["page"],
                    "text": evidence["text"],
                    "confidence": confidence
                }
            )

        validated[
            "constitutional_amendments"
        ][amendment] = {
            "pages": data["pages"],
            "evidence": evidence_records
        }

    return validated


# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

def summarize_confidence(
    validated_provisions: dict
) -> dict:
    """
    Count high, medium and low confidence
    evidence records.
    """

    summary = {
        "high": 0,
        "medium": 0,
        "low": 0
    }

    categories = [
        "articles",
        "sections",
        "constitutional_amendments"
    ]

    for category in categories:

        for data in (
            validated_provisions[
                category
            ].values()
        ):

            for evidence in data[
                "evidence"
            ]:

                confidence = evidence.get(
                    "confidence"
                )

                if confidence in summary:

                    summary[
                        confidence
                    ] += 1

    return summary


# ============================================================
# STEP 27G
# MINIMUM EVIDENCE + RELEVANCE CALIBRATION
# ============================================================

def calculate_provision_relevance(
    provision: str,
    data: dict
) -> float:
    """
    Calculate a calibrated 0-100 relevance score.

    Factors:
        1. Evidence confidence
        2. Evidence frequency
        3. Unique page coverage
        4. Page diversity
        5. Minimum evidence requirement

    The purpose is to distinguish between:

        "The provision was mentioned once."

    and:

        "The provision was repeatedly discussed."

    This is a research ranking heuristic.
    It does not determine actual legal relevance
    and does not provide legal advice.
    """

    pages = data.get(
        "pages",
        []
    )

    evidence = data.get(
        "evidence",
        []
    )

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    evidence_count = len(
        evidence
    )

    page_count = len(
        set(pages)
    )

    if evidence_count == 0:

        return 0.0

    # ========================================================
    # 1. CONFIDENCE QUALITY
    # ========================================================

    high_count = sum(
        1
        for item in evidence
        if item.get(
            "confidence"
        ) == "high"
    )

    medium_count = sum(
        1
        for item in evidence
        if item.get(
            "confidence"
        ) == "medium"
    )

    low_count = sum(
        1
        for item in evidence
        if item.get(
            "confidence"
        ) == "low"
    )

    confidence_quality = (
        high_count * 1.0
        + medium_count * 0.5
        + low_count * 0.1
    ) / evidence_count

    confidence_score = (
        confidence_quality * 100
    )

    # ========================================================
    # 2. EVIDENCE FREQUENCY
    # ========================================================

    # Logarithmic scaling:
    #
    # 1 evidence  -> small score
    # 10 evidence -> moderate score
    # 100 evidence -> high score
    #
    # This prevents extremely frequent provisions
    # from dominating the ranking.

    frequency_score = (
        math.log1p(
            evidence_count
        )
        / math.log1p(1000)
    ) * 100

    frequency_score = min(
        frequency_score,
        100
    )

    # ========================================================
    # 3. PAGE COVERAGE
    # ========================================================

    page_score = (
        math.log1p(
            page_count
        )
        / math.log1p(300)
    ) * 100

    page_score = min(
        page_score,
        100
    )

    # ========================================================
    # 4. PAGE DIVERSITY
    # ========================================================

    if evidence_count > 0:

        page_diversity_ratio = (
            page_count
            / evidence_count
        )

    else:

        page_diversity_ratio = 0.0

    # --------------------------------------------------------
    # A single evidence record should not receive
    # a diversity bonus.
    # --------------------------------------------------------

    if evidence_count == 1:

        diversity_score = 0.0

    else:

        diversity_score = min(
            page_diversity_ratio * 100,
            100
        )

    # ========================================================
    # 5. CORROBORATION SCORE
    # ========================================================

    # More independent evidence records mean
    # stronger corroboration.

    if evidence_count == 1:

        corroboration_score = 0.0

    elif evidence_count == 2:

        corroboration_score = 20.0

    elif evidence_count == 3:

        corroboration_score = 35.0

    elif evidence_count <= 5:

        corroboration_score = 50.0

    elif evidence_count <= 10:

        corroboration_score = 65.0

    elif evidence_count <= 25:

        corroboration_score = 80.0

    elif evidence_count <= 50:

        corroboration_score = 90.0

    else:

        corroboration_score = 100.0

    # ========================================================
    # 6. BASE RELEVANCE SCORE
    # ========================================================

    base_score = (
        confidence_score * 0.35
        + frequency_score * 0.20
        + page_score * 0.25
        + diversity_score * 0.05
        + corroboration_score * 0.15
    )

    # ========================================================
    # 7. MINIMUM EVIDENCE PENALTY
    # ========================================================

    # This is the most important improvement over
    # STEP 27F.
    #
    # A provision appearing once cannot receive a
    # high relevance score merely because its single
    # evidence record has high confidence.

    if evidence_count == 1:

        evidence_multiplier = 0.25

    elif evidence_count == 2:

        evidence_multiplier = 0.40

    elif evidence_count == 3:

        evidence_multiplier = 0.55

    elif evidence_count <= 5:

        evidence_multiplier = 0.70

    elif evidence_count <= 10:

        evidence_multiplier = 0.85

    else:

        evidence_multiplier = 1.0

    calibrated_score = (
        base_score
        * evidence_multiplier
    )

    # ========================================================
    # 8. PAGE COVERAGE BONUS
    # ========================================================

    # Reward provisions discussed across multiple
    # pages, but keep the bonus small.

    if page_count >= 100:

        page_bonus = 5.0

    elif page_count >= 50:

        page_bonus = 3.0

    elif page_count >= 20:

        page_bonus = 2.0

    elif page_count >= 5:

        page_bonus = 1.0

    else:

        page_bonus = 0.0

    calibrated_score += page_bonus

    # ========================================================
    # 9. FINAL SCORE LIMIT
    # ========================================================

    calibrated_score = min(
        max(
            calibrated_score,
            0.0
        ),
        100.0
    )

    return round(
        calibrated_score,
        2
    )


# ============================================================
# RANK LEGAL PROVISIONS
# ============================================================

def rank_legal_provisions(
    validated_provisions: dict,
    top_k: int = 20
) -> list:
    """
    Rank Articles, Sections and Constitutional
    Amendments using the calibrated relevance score.

    Returns the top-k provisions.
    """

    ranked = []

    categories = {
        "articles": "Article",

        "sections": "Section",

        "constitutional_amendments":
            "Constitutional Amendment"
    }

    # ========================================================
    # PROCESS EACH CATEGORY
    # ========================================================

    for category, category_name in (
        categories.items()
    ):

        for provision, data in (
            validated_provisions[
                category
            ].items()
        ):

            relevance_score = (
                calculate_provision_relevance(
                    provision,
                    data
                )
            )

            # =================================================
            # CONFIDENCE COUNTS
            # =================================================

            confidence_counts = {
                "high": 0,
                "medium": 0,
                "low": 0
            }

            for evidence in data.get(
                "evidence",
                []
            ):

                confidence = evidence.get(
                    "confidence"
                )

                if confidence in (
                    confidence_counts
                ):

                    confidence_counts[
                        confidence
                    ] += 1

            # =================================================
            # OVERALL CONFIDENCE
            # =================================================

            total_high = (
                confidence_counts[
                    "high"
                ]
            )

            total_medium = (
                confidence_counts[
                    "medium"
                ]
            )

            if total_high >= 3:

                overall_confidence = (
                    "high"
                )

            elif (
                total_high >= 1
                or total_medium >= 3
            ):

                overall_confidence = (
                    "medium"
                )

            else:

                overall_confidence = (
                    "low"
                )

            # =================================================
            # STORE RESULT
            # =================================================

            ranked.append(
                {
                    "provision": provision,

                    "category":
                        category_name,

                    "relevance_score":
                        relevance_score,

                    "confidence":
                        overall_confidence,

                    "page_count":
                        len(
                            data.get(
                                "pages",
                                []
                            )
                        ),

                    "evidence_count":
                        len(
                            data.get(
                                "evidence",
                                []
                            )
                        ),

                    "confidence_counts":
                        confidence_counts
                }
            )

    # ========================================================
    # SORT
    # ========================================================

    ranked.sort(
        key=lambda item: (
            item[
                "relevance_score"
            ],
            item[
                "page_count"
            ],
            item[
                "evidence_count"
            ]
        ),
        reverse=True
    )

    # ========================================================
    # RETURN TOP K
    # ========================================================

    return ranked[
        :top_k
    ]
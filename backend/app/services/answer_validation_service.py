import re


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower()
    ).strip()


# ============================================================
# REFERENCE SEARCH
# ============================================================

def find_references(
    text: str,
    references: list
) -> list:

    normalized = normalize(text)

    found = []

    for reference in references:

        if normalize(reference) in normalized:
            found.append(reference)

    return found


# ============================================================
# QUESTION TYPE
# ============================================================

def determine_question_type(
    question: str
) -> str:

    q = question.lower()

    if "amendment" in q:
        return "amendments"

    if "article" in q:
        return "articles"

    if (
        "section" in q
        or "sections" in q
    ):
        return "sections"

    return "general"


# ============================================================
# REFERENCES EXPLICITLY REQUESTED BY QUESTION
# ============================================================

def extract_requested_articles(
    question: str,
    available_articles: list
) -> list:

    requested = []

    for article in available_articles:

        article_number = re.search(
            r"article\s+([0-9]+(?:\([0-9A-Za-z]+\))?[A-Za-z]?)",
            article.lower()
        )

        if not article_number:
            continue

        number = article_number.group(1)

        if re.search(
            rf"\barticle\s+{re.escape(number)}\b",
            question,
            flags=re.IGNORECASE
        ):
            requested.append(article)

    return requested


def extract_requested_amendments(
    question: str,
    available_amendments: list
) -> list:

    requested = []

    normalized_question = normalize(
        question
    )

    for amendment in available_amendments:

        normalized_amendment = normalize(
            amendment
        )

        if normalized_amendment in normalized_question:
            requested.append(amendment)

    # If question asks generally about amendments,
    # all evidence-supported amendments are relevant.
    if (
        not requested
        and "amendment" in normalized_question
    ):
        return available_amendments

    return requested


def extract_requested_sections(
    question: str,
    available_sections: list
) -> list:

    requested = []

    for section in available_sections:

        match = re.search(
            r"section\s+([0-9]+(?:\([0-9A-Za-z]+\))?)",
            section.lower()
        )

        if not match:
            continue

        number = match.group(1)

        if re.search(
            rf"\bsection\s+{re.escape(number)}\b",
            question,
            flags=re.IGNORECASE
        ):
            requested.append(section)

    return requested


# ============================================================
# PAGE CITATION DETECTION
# ============================================================

def extract_page_citations(
    answer: str
) -> list:

    return re.findall(
        r"\[Page[s]?\s+\d+(?:,\s*\d+)*\]",
        answer,
        flags=re.IGNORECASE
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def validate_answer(
    question: str,
    answer: str,
    structured_facts: dict
) -> dict:

    question_type = determine_question_type(
        question
    )

    # --------------------------------------------------------
    # Available evidence references
    # --------------------------------------------------------

    amendments = [
        item["name"]
        for item in structured_facts.get(
            "amendments",
            []
        )
    ]

    articles = [
        item["name"]
        for item in structured_facts.get(
            "articles",
            []
        )
    ]

    sections = [
        item["name"]
        for item in structured_facts.get(
            "sections",
            []
        )
    ]

    

    if question_type == "amendments":

        required = extract_requested_amendments(
            question,
            amendments
        )

    elif question_type == "articles":

        required = extract_requested_articles(
            question,
            articles
        )

    elif question_type == "sections":

        required = extract_requested_sections(
            question,
            sections
        )

    else:

        required = []

    

    detected_amendments = find_references(
        answer,
        amendments
    )

    detected_articles = find_references(
        answer,
        articles
    )

    detected_sections = find_references(
        answer,
        sections
    )

   

    if question_type == "amendments":

        detected_required = [
            item
            for item in required
            if item in detected_amendments
        ]

    elif question_type == "articles":

        detected_required = [
            item
            for item in required
            if item in detected_articles
        ]

    elif question_type == "sections":

        detected_required = [
            item
            for item in required
            if item in detected_sections
        ]

    else:

        detected_required = []

    

    missing = [
        item
        for item in required
        if item not in detected_required
    ]

    

    if required:

        coverage = (
            len(detected_required)
            / len(required)
        ) * 100

    else:

        coverage = 100.0

    

    page_citations = extract_page_citations(
        answer
    )

    has_page_citations = (
        len(page_citations) > 0
    )

    

    references_complete = (
        len(missing) == 0
    )

   

    is_complete = (
        references_complete
        and has_page_citations
    )

    

    return {

        "question_type":
            question_type,

        "coverage_score":
            round(
                coverage,
                2
            ),

        "required_references":
            required,

        "detected_required_references":
            detected_required,

        "missing_references":
            missing,

        "detected_amendments":
            detected_amendments,

        "detected_articles":
            detected_articles,

        "detected_sections":
            detected_sections,

        "page_citations":
            page_citations,

        "has_page_citations":
            has_page_citations,

        "references_complete":
            references_complete,

        "is_complete":
            is_complete
    }
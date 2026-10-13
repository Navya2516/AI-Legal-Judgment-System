import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from backend.app.services.llm_service import (
    generate_grounded_answer
)

from backend.app.services.evidence_service import (
    build_evidence_facts
)

from backend.app.services.answer_validation_service import (
    validate_answer
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K_SEMANTIC = 10

TOP_K_FINAL = 5

RERANK_CANDIDATES = 30


# ============================================================
# DOCUMENT METADATA
# ============================================================

def load_judgment_metadata(
    document_id: str
):
    """
    Load structured metadata for the uploaded judgment.
    """

    upload_path = (
        Path("data/processed/uploads")
        / document_id
        / "structured.json"
    )

    document_path = (
        Path("data/processed/documents")
        / document_id
        / "structured.json"
    )

    metadata_path = None

    if upload_path.exists():

        metadata_path = upload_path

    elif document_path.exists():

        metadata_path = document_path

    if metadata_path is None:

        return {}

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return data.get(
            "metadata",
            data
        )

    except Exception:

        return {}


# ============================================================
# METADATA QUERY DETECTION
# ============================================================

def is_metadata_query(
    query: str
) -> bool:
    """
    Detect questions that can be answered from
    structured judgment metadata.
    """

    query_lower = query.lower()

    metadata_terms = [
        "who were the parties",
        "who are the parties",
        "petitioner",
        "respondent",
        "case number",
        "what is the case number",
        "date of judgment",
        "judgment date",
        "when was the judgment",
        "bench",
        "judges",
        "judge",
        "case name",
        "name of the case"
    ]

    return any(
        term in query_lower
        for term in metadata_terms
    )


# ============================================================
# METADATA EVIDENCE
# ============================================================

def build_metadata_evidence(
    metadata: dict
):
    """
    Convert structured metadata into evidence records.
    """

    evidence = []

    petitioner = metadata.get(
        "petitioner",
        metadata.get(
            "petitioner_name"
        )
    )

    respondent = metadata.get(
        "respondent",
        metadata.get(
            "respondent_name"
        )
    )

    case_number = metadata.get(
        "case_number"
    )

    judgment_date = metadata.get(
        "judgment_date",
        metadata.get("date")
    )

    bench = metadata.get(
        "bench",
        metadata.get("judges")
    )

    if petitioner:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_petitioner",
            "score": 1.0,
            "matched_queries": [
                "case parties",
                "petitioner"
            ],
            "text": (
                f"Petitioner: {petitioner}"
            )
        })

    if respondent:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_respondent",
            "score": 1.0,
            "matched_queries": [
                "case parties",
                "respondent"
            ],
            "text": (
                f"Respondent: {respondent}"
            )
        })

    if case_number:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_case_number",
            "score": 1.0,
            "matched_queries": [
                "case number"
            ],
            "text": (
                f"Case Number: {case_number}"
            )
        })

    if judgment_date:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_judgment_date",
            "score": 1.0,
            "matched_queries": [
                "judgment date"
            ],
            "text": (
                f"Judgment Date: {judgment_date}"
            )
        })

    if bench:

        if isinstance(
            bench,
            list
        ):

            bench_text = ", ".join(
                str(item)
                for item in bench
            )

        else:

            bench_text = str(
                bench
            )

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_bench",
            "score": 1.0,
            "matched_queries": [
                "bench",
                "judges"
            ],
            "text": (
                f"Bench / Judges: {bench_text}"
            )
        })

    return evidence


# ============================================================
# LOAD DOCUMENT RESOURCES
# ============================================================

def load_resources(
    document_id: str
):
    """
    Load document-specific FAISS index,
    metadata and embedding model.
    """

    document_dir = (
        Path("data/processed/documents")
        / document_id
    )

    index_path = (
        document_dir
        / "legal_judgment.index"
    )

    metadata_path = (
        document_dir
        / "metadata.json"
    )

    if not index_path.exists():

        raise FileNotFoundError(
            f"FAISS index not found for document: "
            f"{document_id}"
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata not found for document: "
            f"{document_id}"
        )

    print(
        f"Loading FAISS index for document: "
        f"{document_id}"
    )

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Metadata: {len(metadata)}"
    )

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True
    )

    return (
        index,
        metadata,
        model
    )


# ============================================================
# QUERY EXPANSION
# ============================================================
def expand_query(query: str) -> list:
    """
    Expand the user's query with relevant legal terms.

    Expansion is question-aware:
    - Specific amendment questions focus on that amendment.
    - General amendment questions retrieve multiple amendments.
    - Article-specific questions focus on that Article.
    - Parliament/Article 368 questions receive relevant
      constituent-power terminology.

    This remains rule-based and deterministic.
    """

    query_lower = query.lower()

    expanded_queries = [
        query
    ]

    # ========================================================
    # DETECT SPECIFIC AMENDMENT
    # ========================================================

    specific_amendment = None

    amendment_variants = {
        "24th Amendment": [
            "24th amendment",
            "twenty-fourth amendment",
            "twenty fourth amendment",
            "24th constitutional amendment",
        ],
        "25th Amendment": [
            "25th amendment",
            "twenty-fifth amendment",
            "twenty fifth amendment",
            "25th constitutional amendment",
        ],
        "26th Amendment": [
            "26th amendment",
            "twenty-sixth amendment",
            "twenty sixth amendment",
            "26th constitutional amendment",
        ],
        "29th Amendment": [
            "29th amendment",
            "twenty-ninth amendment",
            "twenty ninth amendment",
            "29th constitutional amendment",
        ],
    }

    for canonical, variants in amendment_variants.items():

        if any(
            variant in query_lower
            for variant in variants
        ):
            specific_amendment = canonical
            break

    # ========================================================
    # AMENDMENT QUESTIONS
    # ========================================================

    if specific_amendment:

        # ----------------------------------------------------
        # Specific amendment question
        # ----------------------------------------------------

        expanded_queries.extend([
            specific_amendment,
            f"{specific_amendment} Constitution",
            f"{specific_amendment} Article 13",
            f"{specific_amendment} Article 368",
            f"{specific_amendment} amendment changes",
            f"{specific_amendment} effect",
            f"{specific_amendment} validity",
        ])

    elif (
        "constitutional amendments" in query_lower
        or "constitutional amendment" in query_lower
        or "amendments" in query_lower
        or "amendment" in query_lower
    ):

        # ----------------------------------------------------
        # General amendment question
        # ----------------------------------------------------

        expanded_queries.extend([

            "constitutional amendments considered in the case",
            "constitutional amendments challenged in the case",
            "validity of constitutional amendments",
            "validity of amendments",
            "constitutional amendment Acts",

            "24th Amendment",
            "Twenty-fourth Amendment",

            "25th Amendment",
            "Twenty-fifth Amendment",

            "26th Amendment",
            "Twenty-sixth Amendment",

            "29th Amendment",
            "Twenty-ninth Amendment",
        ])

    # ========================================================
    # BASIC STRUCTURE
    # ========================================================

    if "basic structure" in query_lower:

        expanded_queries.extend([
            "basic structure doctrine",
            "basic structure of the Constitution",
            "basic features of the Constitution",
            "essential features of the Constitution",
            "fundamental features of the Constitution",
            "limitations on constitutional amendment",
            "basic structure cannot be destroyed",
        ])

    # ========================================================
    # PARLIAMENT AMENDMENT POWER
    # ========================================================

    if (
        "parliament" in query_lower
        and "amend" in query_lower
    ):

        expanded_queries.extend([
            "Parliament power to amend Constitution",
            "Article 368 Parliament amendment power",
            "power of Parliament to amend Fundamental Rights",
            "constitutional amendment power",
            "amending power of Parliament",
            "constituent power of Parliament",
            "Parliament constituent power",
        ])

    # ========================================================
    # ARTICLE 368
    # ========================================================

    if "article 368" in query_lower:

        expanded_queries.extend([
            "Article 368",
            "power to amend Constitution",
            "constitutional amendment power",
            "amending power of Parliament",
            "procedure under Article 368",
            "limitations on Article 368",
            "constituent power Article 368",
        ])

    # ========================================================
    # ARTICLE 31C
    # ========================================================

    if "31c" in query_lower:

        expanded_queries.extend([
            "Article 31C",
            "25th Amendment Article 31C",
            "judicial review Article 31C",
            "Directive Principles Article 39",
            "Article 31C amendment",
        ])

    # ========================================================
    # ARTICLE 13
    # ========================================================

    if "article 13" in query_lower:

        expanded_queries.extend([
            "Article 13(2)",
            "law under Article 13",
            "Fundamental Rights constitutional amendment",
            "Article 13 constitutional amendments",
            "Article 13 law made by State",
            "constitutional amendment Article 13",
        ])

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    return list(
        dict.fromkeys(
            expanded_queries
        )
    )




# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    index,
    metadata,
    model,
    top_k=None
):
    """
    Perform semantic similarity search using FAISS.
    """

    if top_k is None:

        top_k = TOP_K_SEMANTIC

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(
        "float32"
    )

    scores, indices = index.search(
        embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        results.append({
            "index_id": int(
                index_id
            ),

            "semantic_score": float(
                score
            ),

            "matched_query": query,

            "chunk_id": metadata[
                index_id
            ][
                "chunk_id"
            ],

            "page_number": metadata[
                index_id
            ][
                "page_number"
            ],

            "text": metadata[
                index_id
            ][
                "text"
            ]
        })

    return results


# ============================================================
# AMENDMENT TERMS
# ============================================================

def amendment_terms():
    """
    Return canonical amendment names and variants.
    """

    return {

        "24th Amendment": [
            "24th amendment",
            "twenty-fourth amendment",
            "twenty fourth amendment",
            "24th constitutional amendment"
        ],

        "25th Amendment": [
            "25th amendment",
            "twenty-fifth amendment",
            "twenty fifth amendment",
            "25th constitutional amendment"
        ],

        "26th Amendment": [
            "26th amendment",
            "twenty-sixth amendment",
            "twenty sixth amendment",
            "26th constitutional amendment"
        ],

        "29th Amendment": [
            "29th amendment",
            "twenty-ninth amendment",
            "twenty ninth amendment",
            "29th constitutional amendment"
        ]
    }


# ============================================================
# FIND AMENDMENT MENTIONS
# ============================================================

def find_amendment_mentions(
    text: str
):
    """
    Detect amendment names in a chunk.
    """

    normalized = text.lower()

    found = []

    for canonical, variants in (
        amendment_terms().items()
    ):

        if any(
            variant in normalized
            for variant in variants
        ):

            found.append(
                canonical
            )

    return found


# ============================================================
# AMENDMENT-AWARE RERANKING
# ============================================================
def amendment_aware_rerank(
    candidates,
    original_query: str,
    final_k: int = TOP_K_FINAL
):
    """
    Amendment-aware reranking.

    Behavior:

    1. If the question explicitly names one amendment,
       prioritize evidence for that amendment only.

    2. If the question asks generally about amendments,
       preserve diversity across amendments.

    3. For non-amendment questions, keep normal ranking.
    """

    query_lower = original_query.lower()

    # ========================================================
    # AMENDMENT VARIANTS
    # ========================================================

    groups = amendment_terms()

    explicitly_requested = []

    for canonical, variants in groups.items():

        if any(
            variant in query_lower
            for variant in variants
        ):
            explicitly_requested.append(
                canonical
            )

    # ========================================================
    # NOT AN AMENDMENT QUESTION
    # ========================================================

    amendment_query = (
        "amendment" in query_lower
        or "amendments" in query_lower
    )

    if not amendment_query:
        return candidates[:final_k]

    # ========================================================
    # SPECIFIC AMENDMENT QUESTION
    # ========================================================

    if explicitly_requested:

        target = explicitly_requested[0]

        target_candidates = []
        other_candidates = []

        for candidate in candidates:

            found = find_amendment_mentions(
                candidate["text"]
            )

            if target in found:
                target_candidates.append(
                    candidate
                )
            else:
                other_candidates.append(
                    candidate
                )

        # Strong evidence for the requested amendment first
        target_candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        selected = target_candidates[
            :final_k
        ]

        # If not enough target evidence exists,
        # fill remaining slots with normal ranking.
        if len(selected) < final_k:

            selected_ids = {
                c["index_id"]
                for c in selected
            }

            for candidate in candidates:

                if len(selected) >= final_k:
                    break

                if candidate["index_id"] in selected_ids:
                    continue

                selected.append(
                    candidate
                )

                selected_ids.add(
                    candidate["index_id"]
                )

        return selected[:final_k]

    # ========================================================
    # GENERAL AMENDMENT QUESTION
    # ========================================================

    selected = []
    selected_ids = set()

    # One strong result for each amendment
    for canonical in groups:

        best_candidate = None

        for candidate in candidates:

            if candidate["index_id"] in selected_ids:
                continue

            amendments_found = (
                find_amendment_mentions(
                    candidate["text"]
                )
            )

            if canonical in amendments_found:

                if (
                    best_candidate is None
                    or candidate["final_score"]
                    > best_candidate["final_score"]
                ):
                    best_candidate = candidate

        if best_candidate is not None:

            selected.append(
                best_candidate
            )

            selected_ids.add(
                best_candidate["index_id"]
            )

    # Fill remaining slots
    for candidate in candidates:

        if len(selected) >= final_k:
            break

        if candidate["index_id"] in selected_ids:
            continue

        selected.append(
            candidate
        )

        selected_ids.add(
            candidate["index_id"]
        )

    return selected[:final_k]





# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve(
    original_query: str,
    index,
    metadata,
    model
):
    """
    Retrieve relevant evidence using:

    1. Query expansion
    2. Semantic search
    3. Multi-query aggregation
    4. Amendment-aware reranking
    """

    queries = expand_query(
        original_query
    )

    all_results = {}

    # --------------------------------------------------------
    # Search every expanded query
    # --------------------------------------------------------

    for query in queries:

        results = semantic_search(
            query,
            index,
            metadata,
            model,
            TOP_K_SEMANTIC
        )

        for result in results:

            index_id = result[
                "index_id"
            ]

            if index_id not in all_results:

                all_results[
                    index_id
                ] = {

                    "index_id":
                        index_id,

                    "best_score":
                        result[
                            "semantic_score"
                        ],

                    "matched_queries": [
                        query
                    ],

                    "chunk_id":
                        result[
                            "chunk_id"
                        ],

                    "page_number":
                        result[
                            "page_number"
                        ],

                    "text":
                        result[
                            "text"
                        ]
                }

            else:

                existing = all_results[
                    index_id
                ]

                if (
                    result[
                        "semantic_score"
                    ]
                    >
                    existing[
                        "best_score"
                    ]
                ):

                    existing[
                        "best_score"
                    ] = result[
                        "semantic_score"
                    ]

                if (
                    query
                    not in
                    existing[
                        "matched_queries"
                    ]
                ):

                    existing[
                        "matched_queries"
                    ].append(
                        query
                    )

    # --------------------------------------------------------
    # Calculate final scores
    # --------------------------------------------------------

    final_results = []

    for result in (
        all_results.values()
    ):

        query_match_bonus = min(
            len(
                result[
                    "matched_queries"
                ]
            ) * 0.03,
            0.15
        )

        result[
            "final_score"
        ] = (
            result[
                "best_score"
            ]
            + query_match_bonus
        )

        final_results.append(
            result
        )

    final_results.sort(
        key=lambda x:
            x["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidates = (
        final_results[
            :RERANK_CANDIDATES
        ]
    )

    # --------------------------------------------------------
    # Amendment reranking
    # --------------------------------------------------------

    final_results = (
        amendment_aware_rerank(
            candidates,
            original_query,
            TOP_K_FINAL
        )
    )

    return (
        queries,
        final_results
    )


# ============================================================
# FORMAT PAGE CITATION
# ============================================================

def format_page_citation(
    pages
) -> str:
    """
    Convert page numbers into deterministic citation text.
    """

    clean_pages = []

    for page in pages:

        if page is None:
            continue

        try:

            page = int(page)

        except (
            TypeError,
            ValueError
        ):

            continue

        if page not in clean_pages:

            clean_pages.append(
                page
            )

    clean_pages.sort()

    if not clean_pages:

        return ""

    if len(clean_pages) == 1:

        return (
            f"[Page {clean_pages[0]}]"
        )

    return (
        "[Pages "
        + ", ".join(
            str(page)
            for page in clean_pages
        )
        + "]"
    )


# ============================================================
# ADD DETERMINISTIC CITATIONS
# ============================================================

def add_deterministic_citation(
    answer: str,
    retrieved_chunks: list
) -> str:
    """
    Add citations from the actual retrieved evidence.

    The LLM never chooses page numbers.
    """

    if not answer:

        return answer

    pages = []

    for chunk in retrieved_chunks:

        page = chunk.get(
            "page_number"
        )

        if page is not None:

            try:

                page = int(page)

            except (
                TypeError,
                ValueError
            ):

                continue

            if page not in pages:

                pages.append(
                    page
                )

    pages.sort()

    if not pages:

        return answer

    # --------------------------------------------------------
    # Split answer into paragraphs
    # --------------------------------------------------------

    paragraphs = [
        p.strip()
        for p in answer.split(
            "\n"
        )
        if p.strip()
    ]

    if not paragraphs:

        return answer

    # --------------------------------------------------------
    # Use a maximum of three strongest evidence pages
    # for normal generated answers.
    # --------------------------------------------------------

    citation_pages = pages[
        :3
    ]

    citation = format_page_citation(
        citation_pages
    )

    final_paragraphs = []

    for paragraph in paragraphs:

        # Don't duplicate an existing citation.
        if re.search(
            r"\[Pages?\s+[0-9,\s]+\]",
            paragraph,
            re.IGNORECASE
        ):

            final_paragraphs.append(
                paragraph
            )

        else:

            final_paragraphs.append(
                f"{paragraph} {citation}"
            )

    return "\n\n".join(
        final_paragraphs
    )


# ============================================================
# DETERMINISTIC REFERENCE ANSWER
# ============================================================

def build_deterministic_reference_answer(
    question: str,
    retrieved_chunks: list
) -> str | None:
    """
    Produce deterministic answers only for questions that explicitly
    ask which legal references are mentioned/involved in the judgment.

    Explanatory questions such as:
        - What did the 24th Amendment change?
        - What was Parliament's power under Article 368?
        - What is the role of Article 13?

    must be handled by the grounded LLM instead.
    """

    question_lower = question.lower()

    facts = build_evidence_facts(
        retrieved_chunks
    )

    # ========================================================
    # AMENDMENT IDENTIFICATION
    # ========================================================

    amendment_identification = (
        "which amendments" in question_lower
        or "which constitutional amendments" in question_lower
        or "what amendments were considered" in question_lower
        or "what constitutional amendments were considered"
        in question_lower
        or "what amendments were mentioned" in question_lower
        or "what constitutional amendments were mentioned"
        in question_lower
        or "what amendments were challenged" in question_lower
        or "what constitutional amendments were challenged"
        in question_lower
        or "amendments mentioned in the case" in question_lower
        or "amendments considered in the case" in question_lower
        or "amendments challenged in the case" in question_lower
    )

    if amendment_identification:

        amendments = facts.get(
            "amendments",
            []
        )

        if not amendments:
            return None

        lines = [
            "The retrieved judgment evidence mentions "
            "the following constitutional amendments:"
        ]

        for item in amendments:

            name = item.get(
                "name",
                "Unknown Amendment"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            if len(pages) == 1:

                page_text = (
                    f"[Page {pages[0]}]"
                )

            else:

                page_text = (
                    "[Pages "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                    + "]"
                )

            lines.append(
                f"- {name} {page_text}"
            )

        return "\n".join(lines)

    # ========================================================
    # ARTICLE IDENTIFICATION
    # ========================================================

    article_identification = (
        "which articles" in question_lower
        or "which constitutional articles" in question_lower
        or "what articles are mentioned" in question_lower
        or "what constitutional articles are mentioned"
        in question_lower
        or "articles mentioned in the case" in question_lower
        or "articles involved in the case" in question_lower
        or "which articles were involved" in question_lower
    )

    if article_identification:

        articles = facts.get(
            "articles",
            []
        )

        if not articles:
            return None

        lines = [
            "The retrieved judgment evidence mentions "
            "the following constitutional Articles:"
        ]

        for item in articles:

            name = item.get(
                "name",
                "Unknown Article"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            if len(pages) == 1:

                page_text = (
                    f"[Page {pages[0]}]"
                )

            else:

                page_text = (
                    "[Pages "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                    + "]"
                )

            lines.append(
                f"- {name} {page_text}"
            )

        return "\n".join(lines)

    # ========================================================
    # NO DETERMINISTIC HANDLER
    # ========================================================

    # Explanatory questions must go to the grounded LLM.
    return None

    # ========================================================
    # AMENDMENT IDENTIFICATION
    # ========================================================

    if (
        "amendment" in question_lower
        and (
            "which" in question_lower
            or "what" in question_lower
            or "considered" in question_lower
            or "mentioned" in question_lower
            or "challenged" in question_lower
        )
    ):

        amendments = facts.get(
            "amendments",
            []
        )

        if not amendments:

            return None

        lines = [
            "The retrieved judgment evidence mentions "
            "the following constitutional amendments:"
        ]

        for item in amendments:

            name = item.get(
                "name",
                "Unknown Amendment"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            citation = (
                format_page_citation(
                    pages
                )
            )

            if citation:

                lines.append(
                    f"- {name} {citation}"
                )

            else:

                lines.append(
                    f"- {name}"
                )

        return "\n".join(
            lines
        )

    # ========================================================
    # ARTICLE IDENTIFICATION
    # ========================================================

    if (
        "article" in question_lower
        and (
            "which" in question_lower
            or "what" in question_lower
            or "mentioned" in question_lower
            or "involved" in question_lower
        )
    ):

        articles = facts.get(
            "articles",
            []
        )

        if not articles:

            return None

        lines = [
            "The retrieved judgment evidence mentions "
            "the following constitutional Articles:"
        ]

        for item in articles:

            name = item.get(
                "name",
                "Unknown Article"
            )

            pages = item.get(
                "evidence_pages",
                []
            )

            citation = (
                format_page_citation(
                    pages
                )
            )

            if citation:

                lines.append(
                    f"- {name} {citation}"
                )

            else:

                lines.append(
                    f"- {name}"
                )

        return "\n".join(
            lines
        )

    return None


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    document_id: str,
    question: str
) -> dict:
    """
    Main RAG question-answering pipeline.

    Flow:

    Question
        ↓
    Load document
        ↓
    Retrieve evidence
        ↓
    Deterministic reference detection
        ↓
    LLM generation if necessary
        ↓
    Deterministic page citation
        ↓
    Answer validation
        ↓
    Final response
    """

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # ========================================================
    # LOAD DOCUMENT RESOURCES
    # ========================================================

    index, metadata, model = (
        load_resources(
            document_id
        )
    )

    # ========================================================
    # RETRIEVE EVIDENCE
    # ========================================================

    (
        expanded_queries,
        retrieved_chunks
    ) = retrieve(
        question,
        index,
        metadata,
        model
    )

    # ========================================================
    # NO EVIDENCE
    # ========================================================

    if not retrieved_chunks:

        return {
            "question": question,
            "expanded_queries":
                expanded_queries,
            "retrieved_evidence": [],
            "answer": (
                "No relevant evidence was found "
                "in the indexed judgment."
            ),
            "validation": {
                "status": "not_validated",
                "reason":
                    "No retrieved evidence."
            }
        }

    # ========================================================
    # STRUCTURED FACTS
    # ========================================================

    structured_facts = (
        build_evidence_facts(
            retrieved_chunks
        )
    )

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    deterministic_answer = (
        build_deterministic_reference_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )
    )

    if deterministic_answer:

        answer = deterministic_answer

    else:

        answer = generate_grounded_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        # ----------------------------------------------------
        # Deterministically add citations.
        # ----------------------------------------------------

        answer = add_deterministic_citation(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )

    # ========================================================
    # VALIDATE ANSWER
    # ========================================================

    validation = validate_answer(
        question=question,
        answer=answer,
        structured_facts=structured_facts
    )

    # ========================================================
    # PREPARE EVIDENCE RESPONSE
    # ========================================================

    evidence = []

    for chunk in retrieved_chunks:

        evidence.append({

            "page_number":
                chunk[
                    "page_number"
                ],

            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "score":
                round(
                    chunk[
                        "final_score"
                    ],
                    4
                ),

            "matched_queries":
                chunk[
                    "matched_queries"
                ],

            "text":
                chunk[
                    "text"
                ]
        })

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "document_id":
            document_id,

        "question":
            question,

        "expanded_queries":
            expanded_queries,

        "retrieved_evidence":
            evidence,

        "answer":
            answer,

        "validation":
            validation
    }
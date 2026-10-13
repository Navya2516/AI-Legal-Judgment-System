import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from backend.app.services.llm_service import (
    generate_grounded_answer
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_SEMANTIC = 10
TOP_K_FINAL = 5


# ============================================================
# LOAD DOCUMENT-SPECIFIC RESOURCES
# ============================================================

def load_resources(document_id: str):
    """
    Load the FAISS index, metadata and embedding model
    belonging to the specified uploaded judgment.
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
            f"FAISS index not found for document: {document_id}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found for document: {document_id}"
        )

    print(
        f"Loading FAISS index for document: {document_id}"
    )

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

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
        MODEL_NAME
    )

    return index, metadata, model


# ============================================================
# LOAD JUDGMENT METADATA
# ============================================================

def load_judgment_metadata(document_id: str) -> dict:
    """
    Load the structured judgment metadata for the document.

    This metadata comes from preprocessing_service.py and
    contains information such as:

    - case number
    - petitioner
    - respondent
    - judgment date
    - bench
    """

    structured_path = (
        Path("data/processed/uploads")
        / document_id
        / "structured.json"
    )

    # The current upload pipeline stores structured.json
    # inside data/processed/uploads/{document_id}/.
    #
    # Also support the current document directory if needed.

    if not structured_path.exists():

        structured_path = (
            Path("data/processed/documents")
            / document_id
            / "structured.json"
        )

    if not structured_path.exists():

        raise FileNotFoundError(
            f"Structured judgment not found for document: {document_id}"
        )

    with open(
        structured_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data.get(
        "metadata",
        {}
    )


# ============================================================
# METADATA QUERY DETECTION
# ============================================================

def is_metadata_query(query: str) -> bool:
    """
    Detect whether the user's question is asking for
    basic judgment metadata rather than legal-content retrieval.
    """

    query_lower = query.lower()

    metadata_terms = [

        # Parties
        "who were the parties",
        "who are the parties",
        "parties in this case",
        "petitioner",
        "respondent",
        "appellant",
        "appellants",
        "respondents",

        # Case number
        "case number",
        "case no",
        "case no.",

        # Date
        "date of judgment",
        "judgment date",
        "when was the judgment",

        # Judges / bench
        "who were the judges",
        "judges in this case",
        "bench",
        "judicial bench",
        "judges",

        # Case identity
        "case name",
        "name of the case"
    ]

    return any(
        term in query_lower
        for term in metadata_terms
    )


# ============================================================
# FORMAT METADATA AS EVIDENCE
# ============================================================

def build_metadata_evidence(
    metadata: dict
) -> list:
    """
    Convert judgment metadata into evidence records that
    can be supplied to the grounded LLM.

    Metadata does not have PDF page numbers, so page_number
    is represented as None.
    """

    evidence = []

    case_number = metadata.get(
        "case_number"
    )

    petitioner = metadata.get(
        "petitioner"
    )

    respondent = metadata.get(
        "respondent"
    )

    judgment_date = metadata.get(
        "judgment_date"
    )

    bench = metadata.get(
        "bench"
    )

    if petitioner or respondent:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_parties",
            "final_score": 1.0,
            "matched_queries": [
                "judgment metadata"
            ],
            "text": (
                f"Petitioner: {petitioner or 'Not available'}\n"
                f"Respondent: {respondent or 'Not available'}"
            )
        })

    if case_number:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_case_number",
            "final_score": 1.0,
            "matched_queries": [
                "judgment metadata"
            ],
            "text": (
                f"Case Number: {case_number}"
            )
        })

    if judgment_date:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_judgment_date",
            "final_score": 1.0,
            "matched_queries": [
                "judgment metadata"
            ],
            "text": (
                f"Date of Judgment: {judgment_date}"
            )
        })

    if bench:

        evidence.append({
            "page_number": None,
            "chunk_id": "metadata_bench",
            "final_score": 1.0,
            "matched_queries": [
                "judgment metadata"
            ],
            "text": (
                f"Bench: {bench}"
            )
        })

    return evidence


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query: str) -> list:
    """
    Expand the user's query with general legal retrieval terms.

    This function intentionally avoids hard-coding a particular
    judgment such as Kesavananda Bharati.
    """

    query_lower = query.lower()

    expanded_queries = [
        query
    ]

    # --------------------------------------------------------
    # Constitutional amendment
    # --------------------------------------------------------

    if (
        "constitutional amendment" in query_lower
        or "constitutional amendments" in query_lower
        or "amend the constitution" in query_lower
        or "amendment power" in query_lower
    ):

        expanded_queries.extend([
            "power to amend the Constitution",
            "constitutional amendment power",
            "validity of constitutional amendment",
            "amending power"
        ])

    # --------------------------------------------------------
    # Parliament amendment power
    # --------------------------------------------------------

    if (
        "parliament" in query_lower
        and (
            "amend" in query_lower
            or "power" in query_lower
        )
    ):

        expanded_queries.extend([
            "Parliament power to amend Constitution",
            "constitutional amending power of Parliament",
            "Article 368 amendment power",
            "power of Parliament to amend"
        ])

    # --------------------------------------------------------
    # Fundamental Rights
    # --------------------------------------------------------

    if (
        "fundamental right" in query_lower
        or "fundamental rights" in query_lower
    ):

        expanded_queries.extend([
            "Fundamental Rights",
            "amendment of Fundamental Rights",
            "abridge Fundamental Rights",
            "constitutional protection of Fundamental Rights"
        ])

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    if "basic structure" in query_lower:

        expanded_queries.extend([
            "basic structure doctrine",
            "basic features of the Constitution",
            "essential features of the Constitution",
            "fundamental features of the Constitution"
        ])

    # --------------------------------------------------------
    # Article 368
    # --------------------------------------------------------

    if "article 368" in query_lower:

        expanded_queries.extend([
            "Article 368",
            "constitutional amendment procedure",
            "power to amend Constitution",
            "amending power"
        ])

    # --------------------------------------------------------
    # Article 13
    # --------------------------------------------------------

    if "article 13" in query_lower:

        expanded_queries.extend([
            "Article 13",
            "Article 13(2)",
            "law under Article 13",
            "constitutional amendment and Article 13"
        ])

    # --------------------------------------------------------
    # Article 31C
    # --------------------------------------------------------

    if "31c" in query_lower:

        expanded_queries.extend([
            "Article 31C",
            "Article 31C constitutional amendment",
            "Directive Principles Article 39",
            "judicial review Article 31C"
        ])

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
    model
):
    """
    Perform semantic similarity search using FAISS.
    """

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        embedding,
        TOP_K_SEMANTIC
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        results.append({
            "index_id": int(index_id),
            "semantic_score": float(score),
            "matched_query": query,
            "chunk_id": metadata[index_id][
                "chunk_id"
            ],
            "page_number": metadata[index_id][
                "page_number"
            ],
            "text": metadata[index_id][
                "text"
            ]
        })

    return results


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
    Retrieve the most relevant evidence chunks.
    """

    queries = expand_query(
        original_query
    )

    all_results = {}

    for query in queries:

        results = semantic_search(
            query,
            index,
            metadata,
            model
        )

        for result in results:

            index_id = result[
                "index_id"
            ]

            if index_id not in all_results:

                all_results[index_id] = {
                    "index_id": index_id,
                    "best_score": result[
                        "semantic_score"
                    ],
                    "matched_queries": [
                        query
                    ],
                    "chunk_id": result[
                        "chunk_id"
                    ],
                    "page_number": result[
                        "page_number"
                    ],
                    "text": result[
                        "text"
                    ]
                }

            else:

                existing = all_results[
                    index_id
                ]

                if (
                    result["semantic_score"]
                    > existing["best_score"]
                ):

                    existing["best_score"] = (
                        result["semantic_score"]
                    )

                if query not in existing[
                    "matched_queries"
                ]:

                    existing[
                        "matched_queries"
                    ].append(query)

    # --------------------------------------------------------
    # Final retrieval score
    # --------------------------------------------------------

    final_results = []

    for result in all_results.values():

        query_match_bonus = min(
            len(
                result["matched_queries"]
            ) * 0.03,
            0.15
        )

        result["final_score"] = (
            result["best_score"]
            + query_match_bonus
        )

        final_results.append(
            result
        )

    final_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return (
        queries,
        final_results[:TOP_K_FINAL]
    )


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    document_id: str,
    question: str
) -> dict:
    """
    Retrieve evidence from the specified judgment
    and generate a grounded answer.

    Metadata-aware:
    - Basic case information comes from structured metadata.
    - Legal-content questions use FAISS retrieval.
    """

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------------
    # Load document resources
    # --------------------------------------------------------

    index, metadata, model = load_resources(
        document_id
    )

    # --------------------------------------------------------
    # Load judgment metadata
    # --------------------------------------------------------

    judgment_metadata = load_judgment_metadata(
        document_id
    )

    # --------------------------------------------------------
    # Metadata-aware retrieval
    # --------------------------------------------------------

    metadata_query = is_metadata_query(
        question
    )

    metadata_evidence = []

    if metadata_query:

        metadata_evidence = build_metadata_evidence(
            judgment_metadata
        )

    # --------------------------------------------------------
    # Legal-content retrieval
    # --------------------------------------------------------

    expanded_queries, retrieved_chunks = retrieve(
        question,
        index,
        metadata,
        model
    )

    # --------------------------------------------------------
    # Combine evidence
    # --------------------------------------------------------

    combined_evidence = []

    if metadata_evidence:

        combined_evidence.extend(
            metadata_evidence
        )

    combined_evidence.extend(
        retrieved_chunks
    )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not combined_evidence:

        return {
            "question": question,
            "expanded_queries": expanded_queries,
            "retrieved_evidence": [],
            "answer": (
                "No relevant evidence was found "
                "in the indexed judgment."
            )
        }

    # --------------------------------------------------------
    # Generate grounded answer
    # --------------------------------------------------------

    answer = generate_grounded_answer(
        question=question,
        retrieved_chunks=combined_evidence
    )

    # --------------------------------------------------------
    # Prepare evidence response
    # --------------------------------------------------------

    evidence = []

    for chunk in combined_evidence:

        evidence.append({
            "page_number": chunk.get(
                "page_number"
            ),
            "chunk_id": chunk.get(
                "chunk_id"
            ),
            "score": round(
                chunk.get(
                    "final_score",
                    1.0
                ),
                4
            ),
            "matched_queries": chunk.get(
                "matched_queries",
                []
            ),
            "text": chunk.get(
                "text",
                ""
            )
        })

    return {
        "question": question,
        "expanded_queries": expanded_queries,
        "retrieved_evidence": evidence,
        "answer": answer
    }
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path(
    "data/processed/embeddings/legal_judgment.index"
)

METADATA_PATH = Path(
    "data/processed/embeddings/metadata.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_SEMANTIC = 10
TOP_K_FINAL = 5


def load_resources():

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
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

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    return index, metadata, model


def normalize_query(query):

    return re.sub(
        r"\s+",
        " ",
        query.lower()
    ).strip()


def extract_legal_terms(query):

    terms = []

    # Constitution Articles
    articles = re.findall(
        r"\barticles?\s+\d+(?:\([a-z0-9]+\))?",
        query,
        flags=re.IGNORECASE
    )

    terms.extend(articles)

    # Sections
    sections = re.findall(
        r"\bsections?\s+\d+(?:\([a-z0-9]+\))?",
        query,
        flags=re.IGNORECASE
    )

    terms.extend(sections)

    # Amendment references
    amendments = re.findall(
        r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|"
        r"fifth|sixth|seventh|eighth|ninth|tenth|eleventh|"
        r"twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
        r"seventeenth|eighteenth|nineteenth|twentieth|"
        r"twenty[- ]first|twenty[- ]second|twenty[- ]third|"
        r"twenty[- ]fourth|twenty[- ]fifth|twenty[- ]sixth|"
        r"twenty[- ]seventh|twenty[- ]eighth|twenty[- ]ninth)"
        r"\s+amendment",
        query,
        flags=re.IGNORECASE
    )

    terms.extend(amendments)

    return list(
        dict.fromkeys(terms)
    )


def semantic_search(
    query,
    index,
    metadata,
    model
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
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
            "keyword_score": 0.0,
            "final_score": float(score),
            "chunk_id": metadata[index_id]["chunk_id"],
            "page_number": metadata[index_id]["page_number"],
            "text": metadata[index_id]["text"]
        })

    return results


def keyword_score(
    query,
    text,
    legal_terms
):

    query_lower = normalize_query(query)
    text_lower = normalize_query(text)

    score = 0.0

    # Exact legal-reference matching
    for term in legal_terms:

        if term.lower() in text_lower:
            score += 0.20

    # Important query words
    query_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            query_lower
        )
    )

    stop_words = {
        "what",
        "which",
        "where",
        "when",
        "does",
        "this",
        "that",
        "case",
        "about",
        "were",
        "involved",
        "court",
        "power",
        "constitution"
    }

    query_words -= stop_words

    matched_words = sum(
        1
        for word in query_words
        if word in text_lower
    )

    if query_words:

        score += (
            matched_words /
            len(query_words)
        ) * 0.20

    return min(
        score,
        0.40
    )


def hybrid_search(
    query,
    index,
    metadata,
    model
):

    legal_terms = extract_legal_terms(
        query
    )

    semantic_results = semantic_search(
        query,
        index,
        metadata,
        model
    )

    for result in semantic_results:

        result["keyword_score"] = keyword_score(
            query,
            result["text"],
            legal_terms
        )

        result["final_score"] = (
            result["semantic_score"] * 0.70
            +
            result["keyword_score"] * 0.30
        )

    semantic_ids = {
        result["index_id"]
        for result in semantic_results
    }

    # ---------------------------------------------------------
    # Keyword scan across metadata
    # ---------------------------------------------------------

    if legal_terms:

        for index_id, chunk in enumerate(
            metadata
        ):

            if index_id in semantic_ids:
                continue

            text = chunk["text"]

            keyword = keyword_score(
                query,
                text,
                legal_terms
            )

            if keyword > 0:

                semantic_results.append({
                    "index_id": index_id,
                    "semantic_score": 0.0,
                    "keyword_score": keyword,
                    "final_score": keyword,
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "text": text
                })

    semantic_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return semantic_results[:TOP_K_FINAL]


def main():

    index, metadata, model = load_resources()

    questions = [
        "What constitutional amendments are involved in this case?",
        "What is the basic structure doctrine?",
        "What did the court decide about Parliament's power to amend the Constitution?"
    ]

    for question_number, question in enumerate(
        questions,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"QUESTION {question_number}"
        )
        print("=" * 70)

        print(
            f"Query: {question}"
        )

        legal_terms = extract_legal_terms(
            question
        )

        if legal_terms:

            print(
                f"Detected legal terms: "
                f"{legal_terms}"
            )

        results = hybrid_search(
            question,
            index,
            metadata,
            model
        )

        print()
        print("TOP HYBRID RESULTS")
        print("-" * 70)

        for rank, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"Rank {rank}"
            )

            print(
                f"Final score    : "
                f"{result['final_score']:.4f}"
            )

            print(
                f"Semantic score : "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"Keyword score  : "
                f"{result['keyword_score']:.4f}"
            )

            print(
                f"Chunk          : "
                f"{result['chunk_id']}"
            )

            print(
                f"Page           : "
                f"{result['page_number']}"
            )

            preview = result["text"][:350]

            print(
                f"Text           : "
                f"{preview}..."
            )


if __name__ == "__main__":
    main()
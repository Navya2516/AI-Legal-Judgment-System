import json
import re
from pathlib import Path

import faiss
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

    print(
        f"Vectors: {index.ntotal}"
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"Metadata: {len(metadata)}"
    )

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    return index, metadata, model


def expand_query(query):
    """
    Add legal terms that are likely to represent
    the same legal concept.

    This is deliberately rule-based for our
    first research baseline.
    """

    query_lower = query.lower()

    expanded_queries = [
        query
    ]

    # ---------------------------------------------------------
    # Constitutional amendments
    # ---------------------------------------------------------

    if (
        "constitutional amendments" in query_lower
        or "amendments" in query_lower
    ):

        expanded_queries.extend([
            "24th Amendment",
            "25th Amendment",
            "29th Amendment",
            "Constitution (Twenty-fourth Amendment) Act",
            "Constitution (Twenty-fifth Amendment) Act",
            "Constitution (Twenty-ninth Amendment) Act",
            "Twenty-fourth Amendment",
            "Twenty-fifth Amendment",
            "Twenty-ninth Amendment"
        ])

    # ---------------------------------------------------------
    # Basic structure doctrine
    # ---------------------------------------------------------

    if "basic structure" in query_lower:

        expanded_queries.extend([
            "basic structure doctrine",
            "basic structure of the Constitution",
            "basic features of the Constitution",
            "limitations on constitutional amendment"
        ])

    # ---------------------------------------------------------
    # Parliament amendment power
    # ---------------------------------------------------------

    if (
        "parliament" in query_lower
        and "amend" in query_lower
    ):

        expanded_queries.extend([
            "Parliament power to amend Constitution",
            "Article 368 Parliament amendment power",
            "power of Parliament to amend Fundamental Rights",
            "constitutional amendment power"
        ])

    # Remove duplicates
    return list(
        dict.fromkeys(expanded_queries)
    )


def semantic_search(
    query,
    index,
    metadata,
    model
):

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
            "chunk_id": metadata[index_id]["chunk_id"],
            "page_number": metadata[index_id]["page_number"],
            "text": metadata[index_id]["text"]
        })

    return results


def retrieve(
    original_query,
    index,
    metadata,
    model
):

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

            index_id = result["index_id"]

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
                    "text": result["text"]
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

    # ---------------------------------------------------------
    # Score based on:
    # 1. Best semantic similarity
    # 2. Number of expanded queries matching
    # ---------------------------------------------------------

    final_results = []

    for result in all_results.values():

        query_match_bonus = min(
            len(result["matched_queries"])
            * 0.03,
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


def main():

    index, metadata, model = load_resources()

    questions = [
        "What constitutional amendments are involved in this case?",
        "What is the basic structure doctrine?",
        "What did the court decide about Parliament's power to amend the Constitution?"
    ]

    for number, question in enumerate(
        questions,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"QUESTION {number}"
        )
        print("=" * 70)

        print(
            f"Original query:\n{question}"
        )

        expanded_queries = expand_query(
            question
        )

        print()
        print("EXPANDED QUERIES")
        print("-" * 70)

        for query in expanded_queries:

            print(
                f"- {query}"
            )

        queries, results = retrieve(
            question,
            index,
            metadata,
            model
        )

        print()
        print("TOP RETRIEVED RESULTS")
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
                f"Final score : "
                f"{result['final_score']:.4f}"
            )

            print(
                f"Best score  : "
                f"{result['best_score']:.4f}"
            )

            print(
                f"Chunk       : "
                f"{result['chunk_id']}"
            )

            print(
                f"Page        : "
                f"{result['page_number']}"
            )

            print(
                f"Matched by  : "
                f"{result['matched_queries']}"
            )

            print(
                f"Text        : "
                f"{result['text'][:350]}..."
            )


if __name__ == "__main__":
    main()
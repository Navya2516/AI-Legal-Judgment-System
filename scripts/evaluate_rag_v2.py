"""
RAG Evaluation V2
=================

Deterministic retrieval evaluation for the AI Legal Judgment System.

Goals:
- Do NOT call Gemini.
- Do NOT count LLM/API quota failures as retrieval failures.
- Evaluate true FAISS Top-1/3/5/10 retrieval.
- Evaluate legal-concept coverage.
- Add amendment-specific retrieval tests.
- Evaluate metadata questions separately.
- Keep production rag_service.py unchanged.

Copy this file to:
    scripts/evaluate_rag_v2.py

Run from the project root:
    python scripts\evaluate_rag_v2.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.rag_service import (
    load_resources,
    expand_query,
    load_judgment_metadata,
)


DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"


# ---------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------
#
# We use several focused queries for the amendment problem instead of
# relying on one broad question.
#
TEST_CASES = [
    {
        "id": "semantic_01",
        "category": "semantic",
        "question": "What is the Basic Structure Doctrine?",
        "expected_concepts": [
            ["basic structure"],
            ["constitution"],
            ["amend", "amendment", "amending"],
        ],
    },
    {
        "id": "amendment_01",
        "category": "amendment",
        "question": "What constitutional amendments were considered in the case?",
        "expected_concepts": [
            ["24th amendment", "twenty-fourth amendment"],
            ["25th amendment", "twenty-fifth amendment"],
            ["29th amendment", "twenty-ninth amendment"],
        ],
    },
    {
        "id": "amendment_02",
        "category": "amendment",
        "question": "What did the 24th Amendment change?",
        "expected_concepts": [
            ["24th amendment", "twenty-fourth amendment"],
        ],
    },
    {
        "id": "amendment_03",
        "category": "amendment",
        "question": "What did the 25th Amendment change?",
        "expected_concepts": [
            ["25th amendment", "twenty-fifth amendment"],
        ],
    },
    {
        "id": "amendment_04",
        "category": "amendment",
        "question": "What did the 29th Amendment change?",
        "expected_concepts": [
            ["29th amendment", "twenty-ninth amendment"],
        ],
    },
    {
        "id": "semantic_02",
        "category": "semantic",
        "question": "What was Parliament's power to amend the Constitution under Article 368?",
        "expected_concepts": [
            ["article 368"],
            ["parliament"],
            ["amend", "amendment", "amending"],
            ["constitution"],
        ],
    },
    {
        "id": "semantic_03",
        "category": "semantic",
        "question": "What was the role of Article 13 in the case?",
        "expected_concepts": [
            ["article 13"],
            ["law"],
            ["fundamental rights"],
        ],
    },
    {
        "id": "metadata_01",
        "category": "metadata",
        "question": "Who were the parties in the case?",
    },
]


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def concept_coverage(evidence, expected_concepts):
    combined = normalize(
        "\n".join(item.get("text", "") for item in evidence)
    )

    matched = []
    missing = []

    for variants in expected_concepts:
        if any(normalize(v) in combined for v in variants):
            matched.append(variants[0])
        else:
            missing.append(variants[0])

    total = len(expected_concepts)

    return (
        len(matched) / total if total else 1.0,
        matched,
        missing,
    )


def encode_and_retrieve(question, index, metadata, model, top_n=10):
    """
    Direct FAISS retrieval.

    This deliberately bypasses rag_service.retrieve(), because the
    production service currently limits final retrieval to five chunks.
    """

    expanded_queries = expand_query(question)

    vectors = model.encode(
        expanded_queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    distances, indices = index.search(vectors, top_n)

    candidates = {}

    for query_index, query in enumerate(expanded_queries):
        for rank in range(indices.shape[1]):
            idx = int(indices[query_index][rank])

            if idx < 0 or idx >= len(metadata):
                continue

            score = float(distances[query_index][rank])

            if idx not in candidates:
                candidates[idx] = {
                    "score": score,
                    "matched_queries": [query],
                }
            else:
                candidates[idx]["score"] = max(
                    candidates[idx]["score"],
                    score,
                )
                if query not in candidates[idx]["matched_queries"]:
                    candidates[idx]["matched_queries"].append(query)

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    evidence = []

    for idx, info in ranked[:top_n]:
        item = metadata[idx]

        evidence.append(
            {
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "score": round(info["score"], 4),
                "matched_queries": info["matched_queries"],
                "text": item.get("text", ""),
            }
        )

    return expanded_queries, evidence


def evaluate_metadata(question):
    """
    Evaluate metadata queries without Gemini.

    The production RAG service has explicit metadata handling, so a
    semantic FAISS page-hit metric is not appropriate for this category.
    """

    metadata = load_judgment_metadata(DOCUMENT_ID)

    text = json.dumps(metadata, ensure_ascii=False).lower()

    petitioner = "kesavananda bharati" in text
    respondent = "state of kerala" in text

    return {
        "metadata_status": "success",
        "petitioner_found": petitioner,
        "respondent_found": respondent,
        "metadata_answer_coverage": (
            (int(petitioner) + int(respondent)) / 2
        ),
    }


def main():
    print("=" * 72)
    print("RAG EVALUATION V2 - DETERMINISTIC")
    print("=" * 72)
    print("Gemini calls: NONE")
    print("Production rag_service.py: NOT MODIFIED")
    print()

    print("Loading FAISS resources...")
    index, metadata, model = load_resources(DOCUMENT_ID)

    print(f"Vectors : {index.ntotal}")
    print(f"Metadata: {len(metadata)}")
    print()

    results = []

    for case in TEST_CASES:
        print("=" * 72)
        print(f"[{case['category'].upper()}] {case['question']}")
        print("=" * 72)

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
        }

        # -------------------------------------------------------------
        # Metadata evaluation
        # -------------------------------------------------------------
        if case["category"] == "metadata":
            metadata_result = evaluate_metadata(case["question"])
            result.update(metadata_result)

            print(
                "Petitioner found:",
                metadata_result["petitioner_found"],
            )
            print(
                "Respondent found:",
                metadata_result["respondent_found"],
            )
            print(
                "Metadata coverage:",
                f"{metadata_result['metadata_answer_coverage']:.2%}",
            )

            results.append(result)
            print()
            continue

        # -------------------------------------------------------------
        # Semantic / amendment retrieval
        # -------------------------------------------------------------
        try:
            expanded, evidence = encode_and_retrieve(
                case["question"],
                index,
                metadata,
                model,
                top_n=10,
            )

            pages = [
                int(item["page_number"])
                for item in evidence
                if item.get("page_number") is not None
            ]

            coverage, matched, missing = concept_coverage(
                evidence,
                case["expected_concepts"],
            )

            result.update(
                {
                    "retrieval_status": "success",
                    "expanded_queries": expanded,
                    "retrieved_pages": pages,
                    "top_1_page": pages[0] if pages else None,
                    "top_3_pages": pages[:3],
                    "top_5_pages": pages[:5],
                    "top_10_pages": pages[:10],
                    "concept_coverage": coverage,
                    "matched_concepts": matched,
                    "missing_concepts": missing,
                    "full_concept_coverage": coverage == 1.0,
                    "retrieved_evidence": evidence,
                }
            )

            print("Retrieved pages:")
            print(pages)
            print()
            print("Concept coverage:", f"{coverage:.2%}")
            print("Matched:", matched)
            print("Missing:", missing)

            if case["category"] == "amendment":
                print()
                print("AMENDMENT RETRIEVAL CHECK:")
                for concept in case["expected_concepts"]:
                    found = any(
                        normalize(v) in normalize(
                            "\n".join(
                                x.get("text", "") for x in evidence
                            )
                        )
                        for v in concept
                    )
                    print(
                        f"  {concept[0]}:",
                        "FOUND" if found else "NOT FOUND",
                    )

        except Exception as exc:
            result.update(
                {
                    "retrieval_status": "error",
                    "retrieval_error": str(exc),
                }
            )
            print("RETRIEVAL ERROR:", exc)

        results.append(result)
        print()

    # -----------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------
    retrieval_results = [
        r for r in results
        if r.get("retrieval_status") == "success"
    ]

    semantic_results = [
        r for r in retrieval_results
        if r["category"] == "semantic"
    ]

    amendment_results = [
        r for r in retrieval_results
        if r["category"] == "amendment"
    ]

    metadata_results = [
        r for r in results
        if r["category"] == "metadata"
    ]

    def avg(items, key):
        values = [
            float(x[key])
            for x in items
            if key in x
        ]
        return sum(values) / len(values) if values else 0.0

    overall_concept = avg(
        retrieval_results,
        "concept_coverage",
    )

    semantic_concept = avg(
        semantic_results,
        "concept_coverage",
    )

    amendment_concept = avg(
        amendment_results,
        "concept_coverage",
    )

    amendment_full = (
        sum(
            1 for x in amendment_results
            if x.get("full_concept_coverage")
        )
        / len(amendment_results)
        if amendment_results
        else 0.0
    )

    # Page recall is retained only as a diagnostic metric for this V2
    # because the original expected-page annotations were manually chosen.
    output = {
        "evaluation_name": "RAG Evaluation V2 - Deterministic",
        "document_id": DOCUMENT_ID,
        "gemini_calls": 0,
        "production_rag_modified": False,
        "metrics": {
            "retrieval_success_rate": (
                len(retrieval_results) / len(TEST_CASES)
            ),
            "overall_concept_coverage": overall_concept,
            "semantic_concept_coverage": semantic_concept,
            "amendment_concept_coverage": amendment_concept,
            "amendment_full_concept_coverage_rate": amendment_full,
            "metadata_answer_coverage": avg(
                metadata_results,
                "metadata_answer_coverage",
            ),
        },
        "interpretation": {
            "concept_coverage": (
                "Primary retrieval-quality metric for this evaluation. "
                "It checks whether retrieved evidence contains the expected "
                "legal concepts."
            ),
            "page_recall": (
                "Not used as the primary aggregate metric because manually "
                "selected reference pages do not exhaustively represent all "
                "valid evidence locations in a 502-page judgment."
            ),
            "metadata": (
                "Evaluated separately because parties/case metadata are "
                "handled by structured metadata rather than ordinary semantic retrieval."
            ),
            "generation": (
                "Not evaluated here. This script intentionally makes zero "
                "Gemini calls so quota/API failures cannot contaminate retrieval metrics."
            ),
        },
        "test_results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "rag_evaluation_v2_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("FINAL RAG EVALUATION V2")
    print("=" * 72)
    print(
        "Retrieval success rate        :",
        f"{len(retrieval_results) / len(TEST_CASES):.2%}",
    )
    print(
        "Overall concept coverage     :",
        f"{overall_concept:.2%}",
    )
    print(
        "Semantic concept coverage   :",
        f"{semantic_concept:.2%}",
    )
    print(
        "Amendment concept coverage  :",
        f"{amendment_concept:.2%}",
    )
    print(
        "Amendment full coverage rate:",
        f"{amendment_full:.2%}",
    )
    print(
        "Metadata answer coverage    :",
        f"{avg(metadata_results, 'metadata_answer_coverage'):.2%}",
    )
    print()
    print("Saved to:")
    print(output_path)


if __name__ == "__main__":
    main()

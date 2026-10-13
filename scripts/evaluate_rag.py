"""
Improved RAG evaluation.

Run this file from the project root after copying it to:
scripts/evaluate_rag.py

It evaluates retrieval separately from Gemini/API generation,
uses direct FAISS Top-10 retrieval, and adds concept coverage.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.rag_service import load_resources, expand_query, answer_question

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

TEST_CASES = [
    {
        "id": "rag_01",
        "question": "What is the Basic Structure Doctrine?",
        "expected_pages": [128, 236, 331, 369],
        "concepts": [
            ["basic structure"],
            ["constitution"],
            ["amend", "amendment", "amending"],
        ],
    },
    {
        "id": "rag_02",
        "question": "What constitutional amendments were considered in the case?",
        "expected_pages": [1, 147, 375, 426, 477],
        "concepts": [
            ["24th amendment", "twenty-fourth amendment"],
            ["25th amendment", "twenty-fifth amendment"],
            ["29th amendment", "twenty-ninth amendment"],
        ],
    },
    {
        "id": "rag_03",
        "question": "What was Parliament's power to amend the Constitution under Article 368?",
        "expected_pages": [71, 121, 331, 375, 388],
        "concepts": [
            ["article 368"],
            ["parliament"],
            ["amend", "amendment", "amending"],
            ["constitution"],
        ],
    },
    {
        "id": "rag_04",
        "question": "What was the role of Article 13 in the case?",
        "expected_pages": [7, 71, 153, 375],
        "concepts": [
            ["article 13"],
            ["law"],
            ["fundamental rights"],
        ],
    },
    {
        "id": "rag_05",
        "question": "Who were the parties in the case?",
        "expected_pages": [1],
        "concepts": [
            ["kesavananda bharati"],
            ["state of kerala"],
        ],
    },
]


def norm(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_top10(question, index, metadata, model):
    expanded = expand_query(question)

    vectors = model.encode(
        expanded,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    distances, indices = index.search(vectors, 10)
    candidates = {}

    for qi, query in enumerate(expanded):
        for rank in range(indices.shape[1]):
            idx = int(indices[qi][rank])
            if idx < 0 or idx >= len(metadata):
                continue

            score = float(distances[qi][rank])

            if idx not in candidates:
                candidates[idx] = {
                    "score": score,
                    "matched_queries": [query],
                }
            else:
                candidates[idx]["score"] = max(
                    candidates[idx]["score"], score
                )
                candidates[idx]["matched_queries"].append(query)

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    evidence = []
    for idx, info in ranked[:10]:
        item = metadata[idx]
        evidence.append({
            "page_number": item.get("page_number"),
            "chunk_id": item.get("chunk_id"),
            "score": round(info["score"], 4),
            "matched_queries": info["matched_queries"],
            "text": item.get("text", ""),
        })

    return expanded, evidence


def concept_coverage(evidence, concepts):
    text = norm("\n".join(x.get("text", "") for x in evidence))
    matched, missing = [], []

    for variants in concepts:
        if any(norm(v) in text for v in variants):
            matched.append(variants[0])
        else:
            missing.append(variants[0])

    return (
        len(matched) / len(concepts) if concepts else 1.0,
        matched,
        missing,
    )


def page_hit(retrieved, expected, k):
    return int(any(p in expected for p in retrieved[:k]))


def citation_metrics(answer, retrieved_pages):
    citations = [int(x) for x in re.findall(r"\[Page\s+(\d+)\]", answer or "")]
    valid = [x for x in citations if x in retrieved_pages]
    return {
        "citation_count": len(citations),
        "valid_citations": len(valid),
        "citation_validity": len(valid) / len(citations) if citations else 0.0,
        "invalid_citations": sorted(set(citations) - set(retrieved_pages)),
    }


def main():
    print("=" * 70)
    print("IMPROVED RAG EVALUATION")
    print("=" * 70)

    index, metadata, model = load_resources(DOCUMENT_ID)
    print(f"Vectors: {index.ntotal}")
    print(f"Metadata: {len(metadata)}")

    results = []
    retrieval_cases = 0
    concept_full = 0
    answer_cases = 0
    api_failures = 0

    for case in TEST_CASES:
        print("\n" + "=" * 70)
        print(case["question"])
        print("=" * 70)

        result = {
            "id": case["id"],
            "question": case["question"],
            "expected_pages": case["expected_pages"],
            "expected_concepts": [x[0] for x in case["concepts"]],
        }

        try:
            expanded, evidence = get_top10(
                case["question"], index, metadata, model
            )

            pages = [
                int(x["page_number"])
                for x in evidence
                if x.get("page_number") is not None
            ]

            result["retrieval_status"] = "success"
            result["expanded_queries"] = expanded
            result["retrieved_pages"] = pages
            result["retrieved_evidence"] = evidence

            result["recall_at_1"] = page_hit(pages, case["expected_pages"], 1)
            result["recall_at_3"] = page_hit(pages, case["expected_pages"], 3)
            result["recall_at_5"] = page_hit(pages, case["expected_pages"], 5)
            result["recall_at_10"] = page_hit(pages, case["expected_pages"], 10)

            overlap = sorted(set(pages) & set(case["expected_pages"]))
            result["expected_page_overlap"] = overlap
            result["evidence_page_hit"] = bool(overlap)

            coverage, matched, missing = concept_coverage(
                evidence, case["concepts"]
            )
            result["concept_coverage"] = coverage
            result["matched_concepts"] = matched
            result["missing_concepts"] = missing

            retrieval_cases += 1
            if coverage == 1.0:
                concept_full += 1

            print("Retrieved pages:", pages)
            print("Page Recall@1 :", result["recall_at_1"])
            print("Page Recall@3 :", result["recall_at_3"])
            print("Page Recall@5 :", result["recall_at_5"])
            print("Page Recall@10:", result["recall_at_10"])
            print("Concept coverage:", f"{coverage:.2%}")
            print("Expected-page overlap:", overlap)

        except Exception as exc:
            result["retrieval_status"] = "error"
            result["retrieval_error"] = str(exc)
            print("RETRIEVAL ERROR:", exc)

        # Generation is evaluated independently.
        if result["retrieval_status"] == "success":
            try:
                generated = answer_question(DOCUMENT_ID, case["question"])
                answer = generated.get("answer", "")
                result["answer_status"] = "success"
                result["answer"] = answer
                result.update(
                    citation_metrics(answer, result["retrieved_pages"])
                )
                answer_cases += 1
                print(
                    "Citation validity:",
                    f"{result['citation_validity']:.2%}",
                )
            except Exception as exc:
                api_failures += 1
                result["answer_status"] = "api_or_generation_error"
                result["answer_error"] = str(exc)
                print("ANSWER/LLM ERROR:", exc)
        else:
            result["answer_status"] = "not_run"

        results.append(result)

    def avg(key):
        vals = [
            r[key] for r in results
            if r.get("retrieval_status") == "success"
            and key in r
        ]
        return sum(vals) / len(vals) if vals else 0.0

    citation_results = [
        r for r in results
        if r.get("answer_status") == "success"
        and "citation_validity" in r
    ]

    citation_avg = (
        sum(r["citation_validity"] for r in citation_results)
        / len(citation_results)
        if citation_results else 0.0
    )

    output = {
        "evaluation_name": "Improved Document-specific RAG Evaluation",
        "document_id": DOCUMENT_ID,
        "total_test_cases": len(TEST_CASES),
        "successful_retrieval_cases": retrieval_cases,
        "successful_answer_cases": answer_cases,
        "api_or_generation_failures": api_failures,
        "metrics": {
            "Page_Recall@1": avg("recall_at_1"),
            "Page_Recall@3": avg("recall_at_3"),
            "Page_Recall@5": avg("recall_at_5"),
            "Page_Recall@10": avg("recall_at_10"),
            "Evidence_Page_Hit_Rate": (
                sum(
                    1 for r in results
                    if r.get("retrieval_status") == "success"
                    and r.get("evidence_page_hit")
                ) / retrieval_cases
                if retrieval_cases else 0.0
            ),
            "Concept_Coverage": avg("concept_coverage"),
            "Full_Concept_Coverage_Rate": (
                concept_full / retrieval_cases if retrieval_cases else 0.0
            ),
            "Citation_Validity": citation_avg,
        },
        "notes": [
            "Page recall uses manually selected reference pages and is not the sole measure of retrieval quality.",
            "Concept coverage checks whether retrieved evidence contains the expected legal concepts.",
            "LLM/API failures are reported separately from retrieval failures.",
            "True Top-10 retrieval is evaluated directly from FAISS because production rag_service currently returns only five final chunks.",
        ],
        "test_results": results,
    }

    output_path = (
        PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("FINAL IMPROVED RAG EVALUATION")
    print("=" * 70)
    print(f"Retrieval success : {retrieval_cases}/{len(TEST_CASES)}")
    print(f"Answer success    : {answer_cases}/{len(TEST_CASES)}")
    print(f"API/LLM failures  : {api_failures}")
    print(f"Page Recall@1     : {avg('recall_at_1'):.2%}")
    print(f"Page Recall@3     : {avg('recall_at_3'):.2%}")
    print(f"Page Recall@5     : {avg('recall_at_5'):.2%}")
    print(f"Page Recall@10    : {avg('recall_at_10'):.2%}")
    print(f"Concept Coverage  : {avg('concept_coverage'):.2%}")
    print(
        f"Full Concept Rate : "
        f"{concept_full / retrieval_cases:.2%}"
        if retrieval_cases else "Full Concept Rate : N/A"
    )
    print(f"Citation Validity : {citation_avg:.2%}")
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()

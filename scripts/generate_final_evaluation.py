import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_DIR = BASE_DIR / "data" / "evaluation"

OUTPUT_FILE = EVALUATION_DIR / "final_evaluation_results.json"


# ============================================================
# RESULT FILES
# ============================================================

FILES = {
    "provision": EVALUATION_DIR / "provision_evaluation_results.json",
    "summary": EVALUATION_DIR / "summary_evaluation_results.json",
    "summary_faithfulness":
        EVALUATION_DIR / "summary_faithfulness_results.json",
    "multilingual_faithfulness":
        EVALUATION_DIR / "multilingual_faithfulness_results.json",
    "multilingual_claim_preservation":
        EVALUATION_DIR / "multilingual_claim_preservation_results.json",
}


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():
        print(f"\nWARNING: File not found:")
        print(path)
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# HELPER
# ============================================================

def get_first(data, *keys, default=None):

    for key in keys:

        if isinstance(data, dict) and key in data:
            return data[key]

    return default


# ============================================================
# ROUGE F1 HELPER
# ============================================================

def get_rouge_f1(data, metric):
    """
    Extract ROUGE F1 from common result-file formats.
    Supports nested {"f1": ...}, {"f1_score": ...}, {"fmeasure": ...},
    or flat keys such as "rouge1_f1".
    """

    aliases = {
        "rouge1": ["rouge1", "rouge_1", "rouge1_f1", "rouge_1_f1"],
        "rouge2": ["rouge2", "rouge_2", "rouge2_f1", "rouge_2_f1"],
        "rougeL": ["rougeL", "rouge_l", "rougeL_f1", "rouge_l_f1"],
    }

    def extract(value):
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, dict):
            for key in ("f1", "f1_score", "fmeasure", "f_measure"):
                if isinstance(value.get(key), (int, float)):
                    return float(value[key])

            scores = value.get("scores")
            if isinstance(scores, dict):
                for key in ("f1", "f1_score", "fmeasure", "f_measure"):
                    if isinstance(scores.get(key), (int, float)):
                        return float(scores[key])

        return None

    for key in aliases.get(metric, []):
        if key in data:
            value = extract(data[key])
            if value is not None:
                return value

    for parent_key in ("rouge", "results", "metrics", "scores"):
        parent = data.get(parent_key) if isinstance(data, dict) else None
        if isinstance(parent, dict):
            for key in aliases.get(metric, []):
                if key in parent:
                    value = extract(parent[key])
                    if value is not None:
                        return value

    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL PROJECT EVALUATION REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load all evaluation files
    # --------------------------------------------------------

    print("\nLoading evaluation results...")

    provision = load_json(FILES["provision"])
    summary = load_json(FILES["summary"])
    summary_faithfulness = load_json(
        FILES["summary_faithfulness"]
    )
    multilingual_faithfulness = load_json(
        FILES["multilingual_faithfulness"]
    )
    multilingual_claim = load_json(
        FILES["multilingual_claim_preservation"]
    )

    # --------------------------------------------------------
    # Extract current evaluation metrics dynamically
    # --------------------------------------------------------

    provision_overall = provision.get(
        "overall",
        {}
    )

    provision_results = provision.get(
        "results",
        {}
    )

    article_results = provision_results.get(
        "articles",
        {}
    )

    amendment_results = provision_results.get(
        "constitutional_amendments",
        {}
    )

    # New evaluator uses target coverage as the primary metric.
    # Fall back to the old strict metrics only if necessary.
    article_target_recall = article_results.get(
        "target_recall",
        article_results.get("recall", 0.0)
    )

    amendment_target_recall = amendment_results.get(
        "target_recall",
        amendment_results.get("recall", 0.0)
    )

    overall_target_recall = provision_overall.get(
        "target_recall",
        provision_overall.get("recall", 0.0)
    )

    # --------------------------------------------------------
    # Build master result
    # --------------------------------------------------------

    final_results = {

        "project": {
            "name":
                "AI-Powered Multilingual Legal Judgment "
                "Explanation System",

            "document_id":
                "0adde2b14334482d83ae091f61003bc6",

            "evaluation_document":
                "Kesavananda Bharati Sripadagalvaru v. "
                "State of Kerala (1973)",

            "evaluation_scope":
                "Baseline component-level evaluation"
        },

        # ====================================================
        # 1. LEGAL PROVISION EXTRACTION
        # ====================================================

        "legal_provision_extraction": {

            "evaluation_method":
                "Key-provision target coverage",

            "methodology_note":
                (
                    "The expected provisions represent key "
                    "provisions for the evaluation judgment "
                    "rather than an exhaustive annotation of "
                    "every provision mentioned in the full "
                    "judgment. Additional extracted provisions "
                    "are therefore reported separately and are "
                    "not treated as false positives in the "
                    "primary evaluation."
                ),

            "articles": {

                "target_count":
                    article_results.get(
                        "target_count",
                        0
                    ),

                "target_found":
                    article_results.get(
                        "target_found",
                        0
                    ),

                "target_missing":
                    article_results.get(
                        "target_missing",
                        0
                    ),

                "target_recall":
                    article_target_recall,

                "predicted_count":
                    article_results.get(
                        "predicted_count",
                        0
                    ),

                "additional_extracted_count":
                    article_results.get(
                        "additional_extracted_count",
                        0
                    ),

                "matched":
                    article_results.get(
                        "matched",
                        []
                    ),

                "missing_targets":
                    article_results.get(
                        "missing_targets",
                        []
                    )
            },

            "constitutional_amendments": {

                "target_count":
                    amendment_results.get(
                        "target_count",
                        0
                    ),

                "target_found":
                    amendment_results.get(
                        "target_found",
                        0
                    ),

                "target_missing":
                    amendment_results.get(
                        "target_missing",
                        0
                    ),

                "target_recall":
                    amendment_target_recall,

                "predicted_count":
                    amendment_results.get(
                        "predicted_count",
                        0
                    ),

                "additional_extracted_count":
                    amendment_results.get(
                        "additional_extracted_count",
                        0
                    ),

                "matched":
                    amendment_results.get(
                        "matched",
                        []
                    ),

                "missing_targets":
                    amendment_results.get(
                        "missing_targets",
                        []
                    )
            },

            "overall": {

                "target_count":
                    provision_overall.get(
                        "target_count",
                        0
                    ),

                "target_found":
                    provision_overall.get(
                        "target_found",
                        0
                    ),

                "target_missing":
                    provision_overall.get(
                        "target_missing",
                        0
                    ),

                "target_recall":
                    overall_target_recall,

                "predicted_count":
                    provision_overall.get(
                        "predicted_count",
                        0
                    ),

                "additional_extracted_count":
                    provision_overall.get(
                        "additional_extracted_count",
                        0
                    ),

                # Retain strict metrics as diagnostics.
                "strict_set_diagnostic":
                    provision_overall.get(
                        "strict_set_diagnostic",
                        {}
                    )
            }
        },

        # ====================================================
        # 2. SUMMARIZATION
        # ====================================================

        "summarization": {

            "rouge1_f1":
                get_rouge_f1(summary, "rouge1"),

            "rouge2_f1":
                get_rouge_f1(summary, "rouge2"),

            "rougeL_f1":
                get_rouge_f1(summary, "rougeL")
        },

        # ====================================================
        # 3. SUMMARY FAITHFULNESS
        # ====================================================

        "summary_faithfulness": {

            "total_claims":
                summary_faithfulness.get(
                    "total_claims",
                    0
                ),

            "claims_with_citations":
                summary_faithfulness.get(
                    "claims_with_citations",
                    0
                ),

            "grounded_claims":
                summary_faithfulness.get(
                    "grounded_claims",
                    0
                ),

            "unsupported_claims":
                summary_faithfulness.get(
                    "unsupported_claims",
                    0
                ),

            "citation_coverage":
                summary_faithfulness.get(
                    "citation_coverage",
                    0.0
                ),

            "grounding_rate":
                summary_faithfulness.get(
                    "grounding_rate",
                    0.0
                ),

            "evaluation_method":
                summary_faithfulness.get(
                    "evaluation_method",
                    "Page-citation based lexical grounding baseline"
                )
        },

        # ====================================================
        # 4. MULTILINGUAL SEMANTIC FAITHFULNESS
        # ====================================================

        "multilingual_semantic_faithfulness": {

            "language":
                multilingual_faithfulness.get(
                    "language",
                    "Tamil"
                ),

            "english_claim_count":
                multilingual_faithfulness.get(
                    "english_claim_count",
                    0
                ),

            "tamil_claim_count":
                multilingual_faithfulness.get(
                    "tamil_claim_count",
                    0
                ),

            "compared_claim_count":
                multilingual_faithfulness.get(
                    "compared_claim_count",
                    0
                ),

            "faithful_claims":
                multilingual_faithfulness.get(
                    "faithful_claims",
                    0
                ),

            "semantic_faithfulness_rate":
                multilingual_faithfulness.get(
                    "semantic_faithfulness_rate",
                    0.0
                ),

            "mean_cross_language_similarity":
                multilingual_faithfulness.get(
                    "mean_cross_language_similarity",
                    0.0
                ),

            "citation_validity":
                multilingual_faithfulness.get(
                    "citation_validity",
                    0.0
                ),

            "citation_coverage":
                multilingual_faithfulness.get(
                    "citation_coverage",
                    0.0
                ),

            "embedding_model":
                multilingual_faithfulness.get(
                    "embedding_model",
                    "paraphrase-multilingual-MiniLM-L12-v2"
                ),

            "faithfulness_threshold":
                multilingual_faithfulness.get(
                    "faithfulness_threshold",
                    0.60
                ),

            "evaluation_method":
                "Cross-language semantic similarity with one-to-one claim matching"
        },

        # ====================================================
        # 5. MULTILINGUAL CLAIM PRESERVATION
        # ====================================================

        "multilingual_claim_preservation": {

            "language":
                multilingual_claim.get(
                    "language",
                    "Tamil"
                ),

            "english_claims":
                multilingual_claim.get(
                    "english_claim_count",
                    0
                ),

            "tamil_claims":
                multilingual_claim.get(
                    "tamil_claim_count",
                    0
                ),

            "compared_claims":
                multilingual_claim.get(
                    "compared_claim_count",
                    0
                ),

            "meaning_preserved":
                multilingual_claim.get(
                    "meaning_preserved",
                    0
                ),

            "meaning_preservation_rate":
                multilingual_claim.get(
                    "meaning_preservation_rate",
                    0.0
                ),

            "minor_issues":
                multilingual_claim.get(
                    "minor_issues",
                    0
                ),

            "major_issues":
                multilingual_claim.get(
                    "major_issues",
                    0
                ),

            "citation_coverage":
                multilingual_claim.get(
                    "citation_coverage",
                    0.0
                ),
            "subject_preserved":
    multilingual_claim.get(
        "dimension_scores",
        {}
    ).get(
        "subject_preserved",
        0.0
    ),

"legal_provision_preserved":
    multilingual_claim.get(
        "dimension_scores",
        {}
    ).get(
        "legal_provision_preserved",
        0.0
    ),

"action_or_power_preserved":
    multilingual_claim.get(
        "dimension_scores",
        {}
    ).get(
        "action_or_power_preserved",
        0.0
    ),

"limitation_or_condition_preserved":
    multilingual_claim.get(
        "dimension_scores",
        {}
    ).get(
        "limitation_or_condition_preserved",
        0.0
    ),

"conclusion_preserved":
    multilingual_claim.get(
        "dimension_scores",
        {}
    ).get(
        "conclusion_preserved",
        0.0
    ),


            "evaluation_type":
                multilingual_claim.get(
                    "evaluation_type",
                    "LLM-assisted legal claim preservation"
                )
        }
    }


    # ========================================================
    # SAVE
    # ========================================================

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 70)
    print("MASTER EVALUATION RESULTS")
    print("=" * 70)

    print("\nLegal Provision Extraction")
    print(
        f"  Target Coverage   : "
        f"{final_results['legal_provision_extraction']['overall']['target_recall'] * 100:.2f}%"
    )
    print(
        f"  Target Provisions : "
        f"{final_results['legal_provision_extraction']['overall']['target_found']}/"
        f"{final_results['legal_provision_extraction']['overall']['target_count']}"
    )
    print(
        f"  Total Extracted   : "
        f"{final_results['legal_provision_extraction']['overall']['predicted_count']}"
    )
    print(
        f"  Additional Found  : "
        f"{final_results['legal_provision_extraction']['overall']['additional_extracted_count']}"
    )

    strict = final_results[
        'legal_provision_extraction'
    ][
        'overall'
    ].get(
        'strict_set_diagnostic',
        {}
    )

    if strict:
        print("\n  Strict Set Diagnostic")
        print(
            f"    Precision       : "
            f"{strict.get('precision', 0.0) * 100:.2f}%"
        )
        print(
            f"    Recall          : "
            f"{strict.get('recall', 0.0) * 100:.2f}%"
        )
        print(
            f"    F1              : "
            f"{strict.get('f1_score', 0.0) * 100:.2f}%"
        )

    print("\nSummarization")
    print(
        f"  ROUGE-1 F1        : "
        f"{final_results['summarization']['rouge1_f1'] * 100:.2f}%"
    )
    print(
        f"  ROUGE-2 F1        : "
        f"{final_results['summarization']['rouge2_f1'] * 100:.2f}%"
    )
    print(
        f"  ROUGE-L F1        : "
        f"{final_results['summarization']['rougeL_f1'] * 100:.2f}%"
    )

    print("\nSummary Faithfulness")
    print(
        f"  Citation Coverage : "
        f"{final_results['summary_faithfulness']['citation_coverage'] * 100:.2f}%"
    )
    print(
        f"  Grounding Rate    : "
        f"{final_results['summary_faithfulness']['grounding_rate'] * 100:.2f}%"
    )

    print("\nTamil Semantic Faithfulness")
    print(
        f"  Semantic Rate     : "
        f"{final_results['multilingual_semantic_faithfulness']['semantic_faithfulness_rate'] * 100:.2f}%"
    )
    print(
        f"  Citation Validity : "
        f"{final_results['multilingual_semantic_faithfulness']['citation_validity'] * 100:.2f}%"
    )
    print(
        f"  Citation Coverage : "
        f"{final_results['multilingual_semantic_faithfulness']['citation_coverage'] * 100:.2f}%"
    )

    print("\nTamil Claim Preservation")
    print(
        f"  Meaning Preserved : "
        f"{final_results['multilingual_claim_preservation']['meaning_preservation_rate'] * 100:.2f}%"
    )
    print(
        f"  Subject           : "
        f"{final_results['multilingual_claim_preservation']['subject_preserved'] * 100:.2f}%"
    )
    print(
        f"  Legal Provision   : "
        f"{final_results['multilingual_claim_preservation']['legal_provision_preserved'] * 100:.2f}%"
    )
    print(
        f"  Action / Power    : "
        f"{final_results['multilingual_claim_preservation']['action_or_power_preserved'] * 100:.2f}%"
    )
    print(
        f"  Limitation        : "
        f"{final_results['multilingual_claim_preservation']['limitation_or_condition_preserved'] * 100:.2f}%"
    )
    print(
        f"  Conclusion        : "
        f"{final_results['multilingual_claim_preservation']['conclusion_preserved'] * 100:.2f}%"
    )

    print("\n" + "-" * 70)

    print("Results saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()
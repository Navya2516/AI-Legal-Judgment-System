import json
from pathlib import Path

from rouge_score import rouge_scorer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "expected_summaries.json"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_summaries"
    / "0adde2b14334482d83ae091f61003bc6_summary.json"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 65)
    print("JUDGMENT SUMMARY EVALUATION")
    print("=" * 65)

    reference_data = load_json(
        REFERENCE_FILE
    )

    generated_data = load_json(
        SUMMARY_FILE
    )

    reference_summary = (
        reference_data[
            "kesavananda_bharati_1973"
        ][
            "reference_summary"
        ]
    )

    generated_summary = (
        generated_data[
            "summary"
        ][
            "simplified_explanation"
        ]
    )

    print("\nReference summary loaded.")
    print("Generated summary loaded.")

    # ========================================================
    # ROUGE
    # ========================================================

    scorer = rouge_scorer.RougeScorer(
        [
            "rouge1",
            "rouge2",
            "rougeL"
        ],
        use_stemmer=True
    )

    scores = scorer.score(
        reference_summary,
        generated_summary
    )

    print("\n" + "=" * 65)
    print("ROUGE RESULTS")
    print("=" * 65)

    results = {}

    for metric, score in scores.items():

        results[metric] = {
            "precision": round(
                score.precision,
                4
            ),
            "recall": round(
                score.recall,
                4
            ),
            "f1": round(
                score.fmeasure,
                4
            )
        }

        print(
            f"\n{metric.upper()}"
        )

        print(
            f"  Precision : "
            f"{score.precision:.4f}"
        )

        print(
            f"  Recall    : "
            f"{score.recall:.4f}"
        )

        print(
            f"  F1        : "
            f"{score.fmeasure:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "summary_evaluation_results.json"
    )

    output = {
        "document_id":
            "0adde2b14334482d83ae091f61003bc6",

        "evaluation_type":
            "Reference-based summarization",

        "metrics":
            results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to:"
    )

    print(output_file)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
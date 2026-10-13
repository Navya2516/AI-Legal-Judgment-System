import sys
import json
from pathlib import Path


# ============================================================
# PROJECT ROOT / PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LEGAL PROVISION SERVICE
# ============================================================

from backend.app.services.legal_provision_service import (
    extract_legal_provisions_with_evidence,
    validate_legal_provisions
)


# ============================================================
# FILE PATHS
# ============================================================

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

STRUCTURED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploads"
    / DOCUMENT_ID
    / "structured.json"
)

EXPECTED_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "expected_sections.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "provision_evaluation_results.json"
)


# ============================================================
# NORMALIZATION
# ============================================================



def normalize(value):
    """
    Normalize legal provision names so that equivalent
    representations are compared correctly.
    """

    value = str(value).strip().lower()

    # Remove extra whitespace
    value = " ".join(value.split())

    # Normalize Article formatting
    value = value.replace("article ", "article ")

    # Article 31-A -> Article 31A
    value = value.replace("article 31-a", "article 31a")

    # Article 31 - A -> Article 31A
    value = value.replace("article 31 - a", "article 31a")

    # Constitutional amendment normalization
    amendment_replacements = {
        "constitution (twenty-fourth amendment) act, 1971":
            "24th amendment",

        "constitution (twenty-fifth amendment) act, 1971":
            "25th amendment",

        "constitution (twenty-ninth amendment) act, 1972":
            "29th amendment",

        "constitution (fourth amendment) act, 1955":
            "4th amendment",

        "constitution (first amendment) act, 1951":
            "1st amendment",

        "constitution (seventeenth amendment) act, 1964":
            "17th amendment",

        "constitution (eighteenth amendment) act, 1966":
            "18th amendment",

        "constitution (tenth amendment) act, 1961":
            "10th amendment",

        "constitution (sixteenth amendment) act, 1963":
            "16th amendment",

        "constitution (twenty-first amendment) act, 1967":
            "21st amendment",

        "constitution (fourteenth amendment) act, 1962":
            "14th amendment",

        "constitution (twenty-sixth amendment) act, 1971":
            "26th amendment",

        "constitution (nineteenth amendment) act, 1966":
            "19th amendment",

        "constitution (fifteenth amendment) act, 1963":
            "15th amendment",

        "constitution (75th amendment) act":
            "75th amendment"
    }

    if value in amendment_replacements:
        return amendment_replacements[value]

    return value


# ============================================================
# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(expected, predicted):
    """
    Evaluate the extractor against a KEY-PROVISION ground truth.

    The expected file contains the provisions that the project
    explicitly identifies as important/relevant for this judgment.
    It is NOT a complete annotation of every Article or amendment
    mentioned anywhere in the 502-page judgment.

    Therefore:
        - Target Recall measures how many expected provisions
          were successfully found.
        - Additional Extracted Provisions are reported separately.
        - The additional provisions are NOT treated as false
          positives because the ground truth is not exhaustive.

    A strict set-comparison metric is also retained as a diagnostic
    value so previous experiments can still be compared.
    """

    expected_set = {
        normalize(item)
        for item in expected
    }

    predicted_set = {
        normalize(item)
        for item in predicted
    }

    # ------------------------------------------------------------
    # Target provisions found
    # ------------------------------------------------------------

    matched_set = (
        expected_set & predicted_set
    )

    missing_set = (
        expected_set - predicted_set
    )

    additional_set = (
        predicted_set - expected_set
    )

    target_count = len(expected_set)
    matched_count = len(matched_set)
    missing_count = len(missing_set)

    if target_count > 0:
        target_recall = (
            matched_count / target_count
        )
    else:
        target_recall = 0.0

    # ------------------------------------------------------------
    # Strict set-comparison diagnostic
    #
    # IMPORTANT:
    # This is NOT the primary project metric because the expected
    # ground truth is intentionally not exhaustive.
    # ------------------------------------------------------------

    strict_true_positive = matched_count
    strict_false_positive = len(additional_set)
    strict_false_negative = missing_count

    if (
        strict_true_positive
        + strict_false_positive
        > 0
    ):
        strict_precision = (
            strict_true_positive
            / (
                strict_true_positive
                + strict_false_positive
            )
        )
    else:
        strict_precision = 0.0

    if (
        strict_true_positive
        + strict_false_negative
        > 0
    ):
        strict_recall = (
            strict_true_positive
            / (
                strict_true_positive
                + strict_false_negative
            )
        )
    else:
        strict_recall = 0.0

    if (
        strict_precision
        + strict_recall
        > 0
    ):
        strict_f1 = (
            2
            * strict_precision
            * strict_recall
            / (
                strict_precision
                + strict_recall
            )
        )
    else:
        strict_f1 = 0.0

    return {
        # Primary evaluation
        "target_count": target_count,
        "target_found": matched_count,
        "target_missing": missing_count,
        "target_recall": target_recall,

        # Extractor breadth
        "predicted_count": len(predicted_set),
        "additional_extracted_count": len(
            additional_set
        ),

        # Primary lists
        "matched": sorted(matched_set),
        "missing_targets": sorted(missing_set),
        "additional_extracted": sorted(
            additional_set
        ),

        # Backward-compatible strict diagnostic
        "strict_set_metrics": {
            "true_positive": strict_true_positive,
            "false_positive": strict_false_positive,
            "false_negative": strict_false_negative,
            "precision": strict_precision,
            "recall": strict_recall,
            "f1_score": strict_f1
        },

        # Backward-compatible aliases.
        # These are retained for older scripts, but the report
        # clearly labels them as strict diagnostics.
        "true_positive": strict_true_positive,
        "false_positive": strict_false_positive,
        "false_negative": strict_false_negative,
        "precision": strict_precision,
        "recall": strict_recall,
        "f1_score": strict_f1,
        "false_positives": sorted(additional_set),
        "false_negatives": sorted(missing_set)
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

# ============================================================

def main():

    print("=" * 60)
    print("LEGAL PROVISION EXTRACTION EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check structured judgment
    # --------------------------------------------------------

    if not STRUCTURED_FILE.exists():
        print("\nERROR:")
        print("Structured judgment file was not found.")
        print(f"Expected location:\n{STRUCTURED_FILE}")
        return

    # --------------------------------------------------------
    # Check ground-truth file
    # --------------------------------------------------------

    if not EXPECTED_FILE.exists():
        print("\nERROR:")
        print("Expected evaluation file was not found.")
        print(f"Expected location:\n{EXPECTED_FILE}")
        return

    # --------------------------------------------------------
    # Load processed judgment
    # --------------------------------------------------------

    try:

        with open(
            STRUCTURED_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            processed_data = json.load(file)

    except Exception as exc:

        print("\nERROR:")
        print("Failed to load structured judgment.")
        print(str(exc))
        return

    pages = processed_data.get("pages", [])

    if not pages:

        print("\nERROR:")
        print("No judgment pages were found.")
        return

    print(
        f"\nJudgment pages loaded: {len(pages)}"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    try:

        with open(
            EXPECTED_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            expected_data = json.load(file)

    except Exception as exc:

        print("\nERROR:")
        print("Failed to load expected evaluation data.")
        print(str(exc))
        return

    if "kesavananda_bharati_1973" not in expected_data:

        print("\nERROR:")
        print(
            "The expected evaluation file does not contain "
            "'kesavananda_bharati_1973'."
        )
        return

    expected = expected_data[
        "kesavananda_bharati_1973"
    ]

    # --------------------------------------------------------
    # Run legal provision extraction
    # --------------------------------------------------------

    print("\nExtracting legal provisions...")
    print("Please wait...")

    try:

        extracted = (
            extract_legal_provisions_with_evidence(
                pages
            )
        )

        validated = (
            validate_legal_provisions(
                extracted
            )
        )

    except Exception as exc:

        print("\nERROR:")
        print("Legal provision extraction failed.")
        print(str(exc))
        return

    # --------------------------------------------------------
    # Validate output structure
    # --------------------------------------------------------

    if not isinstance(validated, dict):

        print("\nERROR:")
        print(
            "Unexpected output from legal provision "
            "validation service."
        )

        print(
            f"Received type: {type(validated)}"
        )

        return

    # --------------------------------------------------------
    # Categories available in the ground truth
    # --------------------------------------------------------

    categories = list(expected.keys())

    # --------------------------------------------------------
    # Convert validated dictionary into predicted lists
    # --------------------------------------------------------

    predicted = {}

    for category in categories:

        category_data = validated.get(
            category,
            {}
        )

        if isinstance(category_data, dict):

            predicted[category] = list(
                category_data.keys()
            )

        elif isinstance(category_data, list):

            predicted[category] = category_data

        else:

            predicted[category] = []

    # --------------------------------------------------------
    # Display expected provisions
    # --------------------------------------------------------

    print("\nEXPECTED PROVISIONS")
    print("-" * 60)

    for category in categories:

        print(f"\n{category}:")

        values = expected.get(
            category,
            []
        )

        if not values:

            print("  None")

        else:

            for value in values:
                print(f"  - {value}")

    # --------------------------------------------------------
    # Display predicted provisions
    # --------------------------------------------------------

    print("\nPREDICTED PROVISIONS")
    print("-" * 60)

    for category in categories:

        print(f"\n{category}:")

        values = predicted.get(
            category,
            []
        )

        if not values:

            print("  None")

        else:

            for value in values:
                print(f"  - {value}")

    # --------------------------------------------------------
    # Calculate evaluation results
    # --------------------------------------------------------

    results = {}

    for category in categories:

        results[category] = calculate_metrics(
            expected.get(category, []),
            predicted.get(category, [])
        )

    # --------------------------------------------------------
    # Display evaluation results
    # --------------------------------------------------------

    print("\nEVALUATION RESULTS")
    print("-" * 60)

    for category in categories:

        metrics = results[category]

        print(
            f"\n{category.upper()}"
        )

        print(
            f"Target Provisions     : "
            f"{metrics['target_count']}"
        )

        print(
            f"Target Provisions Found: "
            f"{metrics['target_found']}"
        )

        print(
            f"Target Provisions Missing: "
            f"{metrics['target_missing']}"
        )

        print(
            f"Target Recall         : "
            f"{metrics['target_recall']:.4f}"
        )

        print(
            f"Total Extracted      : "
            f"{metrics['predicted_count']}"
        )

        print(
            f"Additional Extracted : "
            f"{metrics['additional_extracted_count']}"
        )

        print("\nMatched Target Provisions:")

        if metrics["matched"]:

            for item in metrics["matched"]:
                print(f"  + {item}")

        else:

            print("  None")

        print("\nMissing Target Provisions:")

        if metrics["missing_targets"]:

            for item in metrics["missing_targets"]:
                print(f"  - {item}")

        else:

            print("  None")

        print(
            "\nAdditional Extracted Provisions "
            "(not counted as false positives):"
        )

        if metrics["additional_extracted"]:

            for item in metrics["additional_extracted"]:
                print(f"  * {item}")

        else:

            print("  None")

        # ----------------------------------------------------
        # Strict diagnostic metrics
        # ----------------------------------------------------

        strict = metrics[
            "strict_set_metrics"
        ]

        print(
            "\nStrict Set-Comparison Diagnostic:"
        )

        print(
            f"  Precision: "
            f"{strict['precision']:.4f}"
        )

        print(
            f"  Recall   : "
            f"{strict['recall']:.4f}"
        )

        print(
            f"  F1 Score : "
            f"{strict['f1_score']:.4f}"
        )


# --------------------------------------------------------
    # Overall target coverage
    # --------------------------------------------------------

    total_targets = sum(
        results[category]["target_count"]
        for category in categories
    )

    total_found = sum(
        results[category]["target_found"]
        for category in categories
    )

    total_missing = sum(
        results[category]["target_missing"]
        for category in categories
    )

    total_extracted = sum(
        results[category]["predicted_count"]
        for category in categories
    )

    total_additional = sum(
        results[category][
            "additional_extracted_count"
        ]
        for category in categories
    )

    if total_targets > 0:

        overall_target_recall = (
            total_found / total_targets
        )

    else:

        overall_target_recall = 0.0

    # --------------------------------------------------------
    # Overall strict diagnostic metrics
    # --------------------------------------------------------

    total_tp = sum(
        results[category][
            "strict_set_metrics"
        ]["true_positive"]
        for category in categories
    )

    total_fp = sum(
        results[category][
            "strict_set_metrics"
        ]["false_positive"]
        for category in categories
    )

    total_fn = sum(
        results[category][
            "strict_set_metrics"
        ]["false_negative"]
        for category in categories
    )

    if total_tp + total_fp > 0:

        overall_strict_precision = (
            total_tp
            / (total_tp + total_fp)
        )

    else:

        overall_strict_precision = 0.0

    if total_tp + total_fn > 0:

        overall_strict_recall = (
            total_tp
            / (total_tp + total_fn)
        )

    else:

        overall_strict_recall = 0.0

    if (
        overall_strict_precision
        + overall_strict_recall
        > 0
    ):

        overall_strict_f1 = (
            2
            * overall_strict_precision
            * overall_strict_recall
            / (
                overall_strict_precision
                + overall_strict_recall
            )
        )

    else:

        overall_strict_f1 = 0.0

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Total Target Provisions : "
        f"{total_targets}"
    )

    print(
        f"Total Target Provisions Found: "
        f"{total_found}"
    )

    print(
        f"Total Target Provisions Missing: "
        f"{total_missing}"
    )

    print(
        f"Target Recall / Coverage : "
        f"{overall_target_recall:.4f}"
    )

    print(
        f"Total Provisions Extracted: "
        f"{total_extracted}"
    )

    print(
        f"Additional Extracted Provisions: "
        f"{total_additional}"
    )

    print(
        "\nStrict Set-Comparison Diagnostic"
    )

    print(
        f"  Precision: "
        f"{overall_strict_precision:.4f}"
    )

    print(
        f"  Recall   : "
        f"{overall_strict_recall:.4f}"
    )

    print(
        f"  F1 Score : "
        f"{overall_strict_f1:.4f}"
    )


# --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_data = {
        "document_id": DOCUMENT_ID,

        "document":
            "Kesavananda Bharati v. State of Kerala (1973)",

        "evaluation_type":
            "Legal Provision Extraction",

        "evaluation_method":
            "Key-provision target coverage",

        "methodology_note":
            (
                "The expected provisions represent key/relevant "
                "provisions for this judgment rather than an "
                "exhaustive annotation of every legal provision "
                "mentioned in the full judgment. Additional "
                "extracted provisions are therefore reported "
                "separately and are not counted as false positives "
                "in the primary evaluation."
            ),

        "results": results,

        "overall": {
            "target_count": total_targets,
            "target_found": total_found,
            "target_missing": total_missing,
            "target_recall": overall_target_recall,

            "predicted_count": total_extracted,
            "additional_extracted_count":
                total_additional,

            "strict_set_diagnostic": {
                "true_positive": total_tp,
                "false_positive": total_fp,
                "false_negative": total_fn,
                "precision":
                    overall_strict_precision,
                "recall":
                    overall_strict_recall,
                "f1_score":
                    overall_strict_f1
            },

            # Backward-compatible fields for older
            # evaluation scripts.
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision":
                overall_strict_precision,
            "recall":
                overall_strict_recall,
            "f1_score":
                overall_strict_f1
        },

        "predicted": predicted,

        "expected": expected
    }

    try:

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output_data,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as exc:

        print("\nWARNING:")
        print(
            "Evaluation completed, but the results "
            "could not be saved."
        )
        print(str(exc))
        return

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Evaluation completed successfully.")
    print("Primary metric: key-provision target recall/coverage.")
    print(
        f"Results saved to:\n{OUTPUT_FILE}"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
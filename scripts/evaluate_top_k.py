import json
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT SERVICES
# ============================================================

from backend.app.services.legal_provision_service import (
    extract_legal_provisions_with_evidence,
    validate_legal_provisions,
    rank_legal_provisions
)


# ============================================================
# PATHS
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
    / "top_k_evaluation_results.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):

    value = str(value).strip().lower()

    value = " ".join(value.split())

    # Article normalization
    value = value.replace("article 31-a", "article 31a")
    value = value.replace("article 31 - a", "article 31a")
    value = value.replace("article 31 a", "article 31a")

    # Amendment normalization
    amendment_replacements = {

        "constitution (twenty-fourth amendment) act, 1971":
            "24th amendment",

        "constitution (twenty-fifth amendment) act, 1971":
            "25th amendment",

        "constitution (twenty-ninth amendment) act, 1972":
            "29th amendment",

        "constitution (twenty-fourth amendment act, 1971":
            "24th amendment",

        "constitution (twenty-fifth amendment act, 1971":
            "25th amendment",

        "constitution (twenty-ninth amendment act, 1972":
            "29th amendment",
    }

    if value in amendment_replacements:
        return amendment_replacements[value]

    return value


# ============================================================
# LOAD EXPECTED PROVISIONS
# ============================================================

def load_expected():

    if not EXPECTED_FILE.exists():
        raise FileNotFoundError(
            f"Expected file not found:\n{EXPECTED_FILE}"
        )

    with open(
        EXPECTED_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if "kesavananda_bharati_1973" not in data:

        raise ValueError(
            "Expected file does not contain "
            "'kesavananda_bharati_1973'."
        )

    return data["kesavananda_bharati_1973"]


# ============================================================
# EXTRACT + RANK
# ============================================================

def get_ranked_provisions():

    if not STRUCTURED_FILE.exists():

        raise FileNotFoundError(
            f"structured.json not found:\n{STRUCTURED_FILE}"
        )

    with open(
        STRUCTURED_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        structured_data = json.load(file)

    pages = structured_data.get(
        "pages",
        []
    )

    print(
        f"Judgment pages loaded: {len(pages)}"
    )

    print(
        "\nExtracting legal provisions..."
    )

    provisions = (
        extract_legal_provisions_with_evidence(
            pages
        )
    )

    validated = (
        validate_legal_provisions(
            provisions
        )
    )

    ranked = rank_legal_provisions(
        validated,
        top_k=len(
            validated["articles"]
        )
        + len(
            validated["sections"]
        )
        + len(
            validated["constitutional_amendments"]
        )
    )

    return ranked

# ============================================================
# GET EXPECTED PROVISIONS
# ============================================================

def get_expected(expected_data):

    expected = []

    for category in [
        "articles",
        "sections",
        "constitutional_amendments"
    ]:

        for provision in expected_data.get(
            category,
            []
        ):

            expected.append(
                normalize(provision)
            )

    return expected


# ============================================================
# TOP-K EVALUATION
# ============================================================

def evaluate_top_k(
    ranked,
    expected,
    k
):

    top_k = ranked[:k]

    predicted = [
        normalize(
            item["provision"]
        )
        for item in top_k
    ]

    expected_set = set(expected)

    matched = [
        provision
        for provision in expected_set
        if provision in predicted
    ]

    missing = [
        provision
        for provision in expected_set
        if provision not in predicted
    ]

    recall = (
        len(matched) / len(expected_set)
        if expected_set
        else 0.0
    )

    return {
        "k": k,
        "top_k_provisions": [
            item["provision"]
            for item in top_k
        ],
        "matched": sorted(matched),
        "missing": sorted(missing),
        "matched_count": len(matched),
        "expected_count": len(expected_set),
        "recall_at_k": recall
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("TOP-K LEGAL PROVISION RANKING EVALUATION")
    print("=" * 65)

    expected_data = load_expected()

    expected = get_expected(
        expected_data
    )

    print("\nEXPECTED KEY PROVISIONS")
    print("-" * 65)

    for provision in expected:
        print(
            f"  - {provision}"
        )

    ranked = get_ranked_provisions()

    print("\nTOTAL RANKED PROVISIONS:")
    print(
        f"  {len(ranked)}"
    )

    results = []

    for k in [
        5,
        10,
        20
    ]:

        result = evaluate_top_k(
            ranked,
            expected,
            k
        )

        results.append(result)

        print("\n" + "=" * 65)
        print(
            f"TOP-{k} RESULTS"
        )
        print("=" * 65)

        print("\nRanked provisions:")

        for index, provision in enumerate(
            result["top_k_provisions"],
            start=1
        ):

            print(
                f"  {index:2}. {provision}"
            )

        print("\nMatched:")

        for provision in result["matched"]:
            print(
                f"  + {provision}"
            )

        print("\nMissing:")

        if result["missing"]:

            for provision in result["missing"]:
                print(
                    f"  - {provision}"
                )

        else:

            print(
                "  None"
            )

        print(
            "\nRecall@{} : {:.2%}".format(
                k,
                result["recall_at_k"]
            )
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "document_id": DOCUMENT_ID,
        "expected_provisions": expected,
        "total_ranked_provisions": len(ranked),
        "results": results
    }

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
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
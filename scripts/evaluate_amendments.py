import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from scripts.evaluate_top_k import (
    get_ranked_provisions,
    normalize,
    load_expected
)


def main():

    ranked = get_ranked_provisions()

    expected_data = load_expected()

    expected_amendments = [
        normalize(item)
        for item in expected_data.get(
            "constitutional_amendments",
            []
        )
    ]

    amendment_ranked = [
        item
        for item in ranked
        if item["category"] == "Constitutional Amendment"
    ]

    print("=" * 65)
    print("CONSTITUTIONAL AMENDMENT RANKING EVALUATION")
    print("=" * 65)

    print("\nEXPECTED AMENDMENTS")
    print("-" * 65)

    for item in expected_amendments:
        print(" -", item)

    print("\nRANKED AMENDMENTS")
    print("-" * 65)

    for i, item in enumerate(
        amendment_ranked,
        start=1
    ):

        normalized = normalize(
            item["provision"]
        )

        marker = (
            " <-- EXPECTED"
            if normalized in expected_amendments
            else ""
        )

        print(
            f"{i:3}. "
            f"{item['provision']:<60} "
            f"{item['relevance_score']:6.2f}"
            f"{marker}"
        )

    print("\n" + "=" * 65)

    for k in [1, 3, 5, 10]:

        top_k = amendment_ranked[:k]

        predicted = {
            normalize(item["provision"])
            for item in top_k
        }

        matched = (
            set(expected_amendments)
            & predicted
        )

        recall = (
            len(matched)
            / len(expected_amendments)
            * 100
        )

        print(
            f"Amendment Recall@{k}: "
            f"{recall:.2f}% "
            f"({len(matched)}/{len(expected_amendments)})"
        )


if __name__ == "__main__":
    main()
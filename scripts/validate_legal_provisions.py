import json
from pathlib import Path
from collections import Counter


INPUT_PATH = Path(
    "data/processed/kesavananda_bharati_1973_legal_provisions.json"
)


def main():

    if not INPUT_PATH.exists():
        print(f"File not found: {INPUT_PATH}")
        return

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    provisions = data["provisions"]

    print("=" * 60)
    print("LEGAL PROVISION VALIDATION")
    print("=" * 60)

    print(f"Total unique provisions: {len(provisions)}")
    print()

    # ---------------------------------------------------------
    # 1. Count by type
    # ---------------------------------------------------------

    type_counts = Counter(
        provision["type"]
        for provision in provisions
    )

    print("PROVISIONS BY TYPE")
    print("-" * 60)

    for provision_type, count in type_counts.items():
        print(f"{provision_type:<25}: {count}")

    # ---------------------------------------------------------
    # 2. Constitution Articles
    # ---------------------------------------------------------

    articles = [
        p for p in provisions
        if p["type"] == "Constitution Article"
    ]

    print()
    print("CONSTITUTION ARTICLES")
    print("-" * 60)

    for provision in articles[:20]:

        print(
            f"{provision['reference']:<20}"
            f" | mentions: {provision['mention_count']:<4}"
            f" | pages: {provision['pages'][:5]}"
        )

    if len(articles) > 20:
        print(
            f"... and {len(articles) - 20} more"
        )

    # ---------------------------------------------------------
    # 3. Sections
    # ---------------------------------------------------------

    sections = [
        p for p in provisions
        if p["type"] == "Section"
    ]

    print()
    print("LEGAL SECTIONS")
    print("-" * 60)

    for provision in sections[:20]:

        print(
            f"{provision['reference']:<20}"
            f" | mentions: {provision['mention_count']:<4}"
            f" | pages: {provision['pages'][:5]}"
        )

    if len(sections) > 20:
        print(
            f"... and {len(sections) - 20} more"
        )

    # ---------------------------------------------------------
    # 4. Acts
    # ---------------------------------------------------------

    acts = [
        p for p in provisions
        if p["type"] == "Act"
    ]

    print()
    print("ACTS")
    print("-" * 60)

    for provision in acts[:30]:

        print(
            f"{provision['reference']:<55}"
            f" | mentions: {provision['mention_count']:<3}"
            f" | pages: {provision['pages'][:5]}"
        )

    if len(acts) > 30:
        print(
            f"... and {len(acts) - 30} more"
        )

    # ---------------------------------------------------------
    # 5. Suspicious / incomplete Act names
    # ---------------------------------------------------------

    suspicious = []

    for provision in acts:

        reference = provision["reference"]

        words = reference.split()

        # Very short Act references
        if len(words) <= 2:
            suspicious.append(reference)
            continue

        # Incomplete generic references
        generic_terms = {
            "Amendment Act",
            "Act",
            "Amendment"
        }

        if reference.strip() in generic_terms:
            suspicious.append(reference)

    print()
    print("SUSPICIOUS / INCOMPLETE ACT REFERENCES")
    print("-" * 60)

    if suspicious:

        for reference in suspicious[:30]:
            print(f"- {reference}")

        if len(suspicious) > 30:
            print(
                f"... and {len(suspicious) - 30} more"
            )

    else:
        print("No obvious suspicious Act references found.")

    print()
    print("=" * 60)
    print("VALIDATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
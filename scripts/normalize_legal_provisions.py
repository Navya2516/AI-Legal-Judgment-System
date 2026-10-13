import json
import re
from pathlib import Path
from collections import defaultdict


INPUT_PATH = Path(
    "data/processed/kesavananda_bharati_1973_legal_provisions.json"
)

OUTPUT_PATH = Path(
    "data/processed/kesavananda_bharati_1973_normalized_provisions.json"
)


def normalize_reference(reference: str) -> str:
    """
    Normalize spacing and punctuation in a legal reference.
    """

    reference = re.sub(r"\s+", " ", reference).strip()

    # Normalize comma before year
    reference = re.sub(
        r"Act\s*,\s*(\d{4})",
        r"Act, \1",
        reference
    )

    # Normalize 'Act 1971' -> 'Act, 1971'
    reference = re.sub(
        r"Act\s+(\d{4})",
        r"Act, \1",
        reference
    )

    return reference


def normalize_article(reference: str) -> str:
    """
    Normalize Constitution Article references.
    """

    match = re.search(
        r"(\d+(?:\([a-zA-Z0-9]+\))?)",
        reference
    )

    if match:
        return f"Article {match.group(1)}"

    return reference


def normalize_section(reference: str) -> str:
    """
    Normalize Section references.
    """

    match = re.search(
        r"(\d+(?:\([a-zA-Z0-9]+\))?)",
        reference
    )

    if match:
        return f"Section {match.group(1)}"

    return reference


def create_key(provision_type, reference):

    normalized = normalize_reference(reference)

    if provision_type == "Constitution Article":
        normalized = normalize_article(normalized)

    elif provision_type == "Section":
        normalized = normalize_section(normalized)

    return (
        provision_type,
        normalized.lower()
    )


def main():

    if not INPUT_PATH.exists():

        print(
            f"Input file not found: {INPUT_PATH}"
        )

        return

    print(
        "Loading extracted provisions..."
    )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    normalized = {}

    for provision in data["provisions"]:

        provision_type = provision["type"]
        reference = provision["reference"]

        key = create_key(
            provision_type,
            reference
        )

        normalized_reference = (
            normalize_article(reference)
            if provision_type == "Constitution Article"
            else normalize_section(reference)
            if provision_type == "Section"
            else normalize_reference(reference)
        )

        if key not in normalized:

            normalized[key] = {
                "type": provision_type,
                "reference": normalized_reference,
                "mention_count": 0,
                "pages": [],
                "evidence": []
            }

        target = normalized[key]

        target["mention_count"] += provision[
            "mention_count"
        ]

        # Merge pages
        for page in provision["pages"]:

            if page not in target["pages"]:
                target["pages"].append(page)

        # Merge evidence
        existing_evidence = {
            (
                item["page"],
                item["text"]
            )
            for item in target["evidence"]
        }

        for item in provision["evidence"]:

            evidence_key = (
                item["page"],
                item["text"]
            )

            if evidence_key not in existing_evidence:

                if len(target["evidence"]) < 5:

                    target["evidence"].append(
                        item
                    )

    # Sort pages
    for provision in normalized.values():

        provision["pages"] = sorted(
            provision["pages"]
        )

        provision["evidence"] = sorted(
            provision["evidence"],
            key=lambda x: x["page"]
        )

    normalized_list = sorted(
        normalized.values(),
        key=lambda x: (
            x["type"],
            x["reference"]
        )
    )

    result = {
        "source_file": data["source_file"],
        "total_pages": data["total_pages"],
        "original_unique_provisions": data[
            "unique_provisions"
        ],
        "normalized_unique_provisions": len(
            normalized_list
        ),
        "provisions": normalized_list
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("LEGAL PROVISION NORMALIZATION COMPLETED")
    print("=" * 60)

    print(
        f"Original unique provisions : "
        f"{data['unique_provisions']}"
    )

    print(
        f"Normalized provisions      : "
        f"{len(normalized_list)}"
    )

    print(
        f"Output file                : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Summary by type
    # ---------------------------------------------------------

    counts = defaultdict(int)

    for provision in normalized_list:
        counts[provision["type"]] += 1

    print()
    print("NORMALIZED COUNTS")
    print("-" * 60)

    for provision_type, count in counts.items():

        print(
            f"{provision_type:<25}: {count}"
        )

    # ---------------------------------------------------------
    # Show normalized duplicate examples
    # ---------------------------------------------------------

    print()
    print("SAMPLE NORMALIZED ACTS")
    print("-" * 60)

    acts = [
        p for p in normalized_list
        if p["type"] == "Act"
    ]

    for act in acts[:20]:

        print(
            f"- {act['reference']} "
            f"(mentions: {act['mention_count']})"
        )

    print()
    print("=" * 60)
    print("NORMALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
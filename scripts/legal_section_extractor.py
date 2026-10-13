import json
import re
import sys
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize PDF line breaks and whitespace."""

    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_articles(text: str):
    """Extract Constitution Article references."""

    results = []

    pattern = r"\bArticles?\s+([^.;:\n]+)"

    for match in re.finditer(
        pattern,
        text,
        flags=re.IGNORECASE
    ):

        content = match.group(1)

        content = re.split(
            r"\b(?:of|to|under|and\s+the|which|as|was|is|are)\b",
            content,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        article_numbers = re.findall(
            r"\d+(?:\([a-zA-Z0-9]+\))?",
            content
        )

        for number in article_numbers:

            results.append({
                "type": "Constitution Article",
                "reference": f"Article {number}"
            })

    return results


def extract_sections(text: str):
    """Extract legal Section references."""

    results = []

    pattern = r"\bSections?\s+(\d+(?:\([a-zA-Z0-9]+\))?)"

    for match in re.finditer(
        pattern,
        text,
        flags=re.IGNORECASE
    ):

        results.append({
            "type": "Section",
            "reference": f"Section {match.group(1)}"
        })

    return results


def extract_acts(text: str):
    """Extract likely Indian Acts."""

    results = []

    patterns = [
        r"\b(?:Constitution\s*)?\([A-Za-z0-9\s\-]+Amendment\)\s+Act,?\s+\d{4}",
        r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,8}\s+Act,?\s+\d{4}",
        r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,8}\s+Act\s+\d{4}"
    ]

    for pattern in patterns:

        for match in re.finditer(pattern, text):

            reference = normalize_text(match.group(0))

            if len(reference.split()) >= 3:

                results.append({
                    "type": "Act",
                    "reference": reference
                })

    return results


def extract_provisions_from_page(page_text: str):

    text = normalize_text(page_text)

    provisions = []

    provisions.extend(extract_articles(text))
    provisions.extend(extract_sections(text))
    provisions.extend(extract_acts(text))

    return provisions


def add_evidence(
    evidence_map,
    provision,
    page_number,
    page_text
):

    key = (
        provision["type"],
        provision["reference"].lower()
    )

    if key not in evidence_map:

        evidence_map[key] = {
            "type": provision["type"],
            "reference": provision["reference"],
            "mention_count": 0,
            "pages": [],
            "evidence": []
        }

    record = evidence_map[key]

    record["mention_count"] += 1

    if page_number not in record["pages"]:
        record["pages"].append(page_number)

    if len(record["evidence"]) < 5:

        cleaned_page = normalize_text(page_text)

        position = cleaned_page.lower().find(
            provision["reference"].lower()
        )

        if position >= 0:

            start = max(0, position - 150)
            end = min(
                len(cleaned_page),
                position + 350
            )

            snippet = cleaned_page[start:end]

            record["evidence"].append({
                "page": page_number,
                "text": snippet
            })


def main():

    # --------------------------------------------------
    # INPUT FROM COMMAND LINE
    # --------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python scripts\\legal_section_extractor.py "
            "<structured_json>"
        )

        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():

        print(
            f"Input file not found: {input_path}"
        )

        return

    # Automatically create output filename
    output_path = input_path.with_name(
        input_path.stem.replace(
            "_structured",
            ""
        ) + "_legal_provisions.json"
    )

    print(
        f"Loading structured judgment:\n{input_path}"
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        judgment = json.load(file)

    evidence_map = {}

    total_mentions = 0

    for page in judgment["pages"]:

        page_number = page["page_number"]
        page_text = page["text"]

        provisions = extract_provisions_from_page(
            page_text
        )

        total_mentions += len(provisions)

        for provision in provisions:

            add_evidence(
                evidence_map,
                provision,
                page_number,
                page_text
            )

    unique_provisions = sorted(
        evidence_map.values(),
        key=lambda x: (
            x["type"],
            x["reference"]
        )
    )

    result = {
        "source_file": judgment["source_file"],
        "total_pages": judgment["total_pages"],
        "total_mentions": total_mentions,
        "unique_provisions": len(unique_provisions),
        "provisions": unique_provisions
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
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
    print("LEGAL PROVISION EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"Pages processed   : "
        f"{judgment['total_pages']}"
    )

    print(
        f"Total mentions    : "
        f"{total_mentions}"
    )

    print(
        f"Unique provisions : "
        f"{len(unique_provisions)}"
    )

    print(
        f"Output file       : "
        f"{output_path}"
    )

    print("=" * 60)

    print()
    print("Sample provisions:")

    for provision in unique_provisions[:15]:

        print(
            f"- {provision['type']}: "
            f"{provision['reference']} "
            f"(mentions: "
            f"{provision['mention_count']}, "
            f"pages: "
            f"{provision['pages'][:5]})"
        )


if __name__ == "__main__":
    main()
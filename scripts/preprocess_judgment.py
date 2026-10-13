import json
import re
import sys
from pathlib import Path


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"')
    text = text.replace("â€\x9d", '"')

    lines = [line.rstrip() for line in text.splitlines()]

    cleaned = []
    previous_blank = False

    for line in lines:
        if not line.strip():
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
        else:
            cleaned.append(line)
            previous_blank = False

    return "\n".join(cleaned).strip()


def split_pages(text: str):
    """
    Supports page markers if they exist.
    If not present, treats the document as one logical page.
    """

    pattern = r"--- PAGE (\d+) ---"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [{
            "page_number": 1,
            "text": text.strip()
        }]

    pages = []

    for i, match in enumerate(matches):

        page_number = int(match.group(1))
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        page_text = text[start:end].strip()

        pages.append({
            "page_number": page_number,
            "text": page_text
        })

    return pages

def extract_metadata(text: str):

    metadata = {
        "case_number": None,
        "petitioner": None,
        "respondent": None,
        "judgment_date": None,
        "bench": None
    }

    # Case number
    patterns = [
        r"(CIVIL APPEAL NO\.\s*[^\n]+)",
        r"(CRIMINAL APPEAL NO\.\s*[^\n]+)",
        r"(WRIT PETITION\s*\([^)]+\)\s*NO\.\s*[^\n]+)",
        r"(CASE NO\.?:\s*[^\n]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["case_number"] = " ".join(
                match.group(1).split()
            )
            break

    # Petitioner
    match = re.search(
        r"\n\s*([A-Z][A-Z .&'-]{2,})\s*\n\s*(?:…|\.\.\.)?\s*Appellant",
        text
    )

    if match:
        metadata["petitioner"] = " ".join(
            match.group(1).split()
        )

    # Respondent
    match = re.search(
        r"Versus\s*\n\s*([A-Z][A-Z .&'-]{2,})",
        text,
        re.IGNORECASE
    )

    if match:
        metadata["respondent"] = " ".join(
            match.group(1).split()
        )

    # Judgment date
    date_patterns = [
        r"Date\s*[:\-]\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
        r"DATE OF JUDGMENT\s*[:\-]\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
        r"Date\s*:\s*(\d{1,2}\.\d{1,2}\.\d{4})"
    ]

    for pattern in date_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            metadata["judgment_date"] = match.group(1)
            break

    # Bench / Coram
    bench_patterns = [
        r"Coram\s*:\s*([^\n]+)",
        r"CORAM\s*:\s*([^\n]+)"
    ]

    for pattern in bench_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            metadata["bench"] = " ".join(
                match.group(1).split()
            )
            break

    return metadata



def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/preprocess_judgment.py "
            "<text_file>"
        )
        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    output_path = input_path.with_name(
        input_path.stem + "_structured.json"
    )

    print("Reading extracted judgment:")
    print(input_path)

    raw_text = input_path.read_text(
        encoding="utf-8"
    )

    cleaned_text = clean_text(raw_text)

    pages = split_pages(cleaned_text)

    if len(pages) == 1:
        print("Warning: No page markers detected.")

    metadata = extract_metadata(
        pages[0]["text"]
    )

    structured_data = {

        "document_type": "Indian Court Judgment",

        "source_file": input_path.name,

        "total_pages": len(pages),

        "metadata": metadata,

        "pages": pages
    }

    output_path.write_text(
        json.dumps(
            structured_data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("JUDGMENT PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Pages processed : {len(pages)}")
    print(f"Output file     : {output_path}")

    print()
    print("Metadata:")

    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    legal content and paragraph structure.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Fix common encoding artifacts
    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"')
    text = text.replace("â€\x9d", '"')

    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        if not line.strip():
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


def clean_pages(pages: list) -> list:
    """
    Clean the text of every extracted page.
    """

    cleaned_pages = []

    for page in pages:
        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": clean_text(page["text"])
        })

    return cleaned_pages


def extract_metadata(first_page: str):
    """
    Extract basic judgment metadata from the first page.
    """

    metadata = {
        "case_number": None,
        "petitioner": None,
        "respondent": None,
        "judgment_date": None,
        "bench": None
    }

    patterns = {
        "case_number": r"CASE NO\.:\s*(.*?)\nPETITIONER:",
        "petitioner": r"PETITIONER:\s*(.*?)\nRESPONDENT:",
        "respondent": r"RESPONDENT:\s*(.*?)\nDATE OF JUDGMENT:",
        "judgment_date": r"DATE OF JUDGMENT:\s*(.*?)\nBENCH:",
        "bench": r"BENCH:\s*(.*?)\nJUDGMENT:"
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            first_page,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            metadata[key] = " ".join(
                match.group(1).split()
            )

    return metadata


def preprocess_judgment(extracted_data: dict) -> dict:
    """
    Preprocess extracted judgment data and extract metadata.
    """

    pages = extracted_data.get("pages", [])

    if not pages:
        raise ValueError("No pages found in extracted judgment.")

    cleaned_pages = clean_pages(pages)

    metadata = extract_metadata(
        cleaned_pages[0]["text"]
    )

    return {
        "document_type": "Indian Court Judgment",
        "source_file": extracted_data.get("source_file"),
        "total_pages": len(cleaned_pages),
        "metadata": metadata,
        "pages": cleaned_pages
    }
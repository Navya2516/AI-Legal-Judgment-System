from pathlib import Path
import pymupdf


def extract_pdf_text(pdf_path: str):
    """
    Extract text from every page of a PDF.

    Returns a list containing page number and extracted text.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()

    return {
        "source_file": pdf_path.name,
        "total_pages": len(pages),
        "pages": pages
    }
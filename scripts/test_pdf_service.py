import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.pdf_service import extract_pdf_text


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "uploads" / "kesavananda_bharati_1973.pdf"


def main():
    print("=" * 60)
    print("PDF SERVICE TEST")
    print("=" * 60)

    result = extract_pdf_text(str(PDF_PATH))

    print(f"File        : {result['source_file']}")
    print(f"Total pages : {result['total_pages']}")

    print("\nFirst page preview:")
    print("-" * 60)
    print(result["pages"][0]["text"][:1000])

    print("\n" + "=" * 60)
    print("PDF TEXT EXTRACTION SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()
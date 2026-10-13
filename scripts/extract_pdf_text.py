import sys
from pathlib import Path
import fitz


def extract_pdf_text(pdf_path, output_path):
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"Reading: {pdf_path}")

    doc = fitz.open(pdf_path)

    text = ""

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += page_text + "\n"

        if page_num % 20 == 0:
            print(f"Processed {page_num}/{len(doc)} pages")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print("\nPDF text extraction completed.")
    print(f"Output saved to: {output_path}")
    print(f"Pages extracted: {len(doc)}")
    print(f"Characters extracted: {len(text)}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts\\extract_pdf_text.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        pdf_name = Path(pdf_path).stem
        output_path = f"data/processed/{pdf_name}.txt"

    extract_pdf_text(pdf_path, output_path)
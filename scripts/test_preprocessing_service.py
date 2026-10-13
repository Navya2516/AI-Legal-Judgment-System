import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.pdf_service import extract_pdf_text
from backend.app.services.preprocessing_service import preprocess_judgment


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uploads"
    / "853b3b787bb04672bea14df88e68f5bb_kesavananda_bharati_1973.pdf"
)


def main():
    print("=" * 60)
    print("PREPROCESSING SERVICE TEST")
    print("=" * 60)

    print("\n1. Extracting PDF text...")

    extracted_data = extract_pdf_text(
        str(PDF_PATH)
    )

    print(
        f"   Pages extracted: "
        f"{extracted_data['total_pages']}"
    )

    print("\n2. Preprocessing judgment...")

    processed_data = preprocess_judgment(
        extracted_data
    )

    print(
        f"   Pages processed: "
        f"{processed_data['total_pages']}"
    )

    print("\n3. Extracted metadata:")
    print("-" * 60)

    for key, value in processed_data["metadata"].items():
        print(f"{key}: {value}")

    print("\n4. First page preview:")
    print("-" * 60)

    print(
        processed_data["pages"][0]["text"][:1000]
    )

    print("\n5. Testing JSON serialization...")

    json.dumps(
        processed_data,
        ensure_ascii=False
    )

    print("   JSON serialization successful.")

    print("\n" + "=" * 60)
    print("PREPROCESSING SERVICE TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()
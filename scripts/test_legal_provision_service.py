import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.pdf_service import extract_pdf_text
from backend.app.services.preprocessing_service import preprocess_judgment
from backend.app.services.legal_provision_service import (
    extract_legal_provisions_with_evidence,
    validate_legal_provisions,
    summarize_confidence,
    rank_legal_provisions
)

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "judgments"
    / "kesavananda_bharati_1973.pdf"
)

print("=" * 60)
print("LEGAL PROVISION EXTRACTION TEST")
print("=" * 60)

# --------------------------------------------------
# 1. Extract PDF text
# --------------------------------------------------

print("\n1. Extracting PDF text...")

extracted_data = extract_pdf_text(
    str(PDF_PATH)
)

print(
    f"   Pages extracted: "
    f"{extracted_data['total_pages']}"
)

# --------------------------------------------------
# 2. Preprocess judgment
# --------------------------------------------------

print("\n2. Preprocessing judgment...")

processed_data = preprocess_judgment(
    extracted_data
)

print(
    f"   Pages processed: "
    f"{processed_data['total_pages']}"
)

# --------------------------------------------------
# 3. Extract legal provisions
# --------------------------------------------------

print("\n3. Extracting legal provisions...")

provisions = extract_legal_provisions_with_evidence(
    processed_data["pages"]
)

# --------------------------------------------------
# 4. Display extracted provisions
# --------------------------------------------------

print("\n" + "-" * 60)

print("\nCONSTITUTION ARTICLES:")

for article, data in provisions["articles"].items():
    print(
        f"  {article} "
        f"-> {len(data['pages'])} page(s), "
        f"{len(data['evidence'])} evidence record(s)"
    )

print("\nLEGAL SECTIONS:")

for section, data in provisions["sections"].items():
    print(
        f"  {section} "
        f"-> {len(data['pages'])} page(s), "
        f"{len(data['evidence'])} evidence record(s)"
    )

print("\nCONSTITUTIONAL AMENDMENTS:")

for amendment, data in (
    provisions["constitutional_amendments"].items()
):
    print(
        f"  {amendment} "
        f"-> {len(data['pages'])} page(s), "
        f"{len(data['evidence'])} evidence record(s)"
    )

# --------------------------------------------------
# 5. Validate legal provisions
# --------------------------------------------------

print("\n" + "-" * 60)

print("\n4. Validating legal provisions...")

validated_provisions = validate_legal_provisions(
    provisions
)

print("   Validation completed.")

# --------------------------------------------------
# 6. Confidence summary
# --------------------------------------------------

confidence_summary = summarize_confidence(
    validated_provisions
)

print("\nCONFIDENCE SUMMARY:")

print(
    f"  High confidence: "
    f"{confidence_summary['high']}"
)

print(
    f"  Medium confidence: "
    f"{confidence_summary['medium']}"
)

print(
    f"  Low confidence: "
    f"{confidence_summary['low']}"
)

# --------------------------------------------------
# 7. Rank legal provisions
# --------------------------------------------------

print("\n" + "-" * 60)

print("\n5. Ranking legal provisions...")

ranked_provisions = rank_legal_provisions(
    validated_provisions,
    top_k=20
)

print("   Ranking completed.")

# --------------------------------------------------
# 8. Display top 20 ranked provisions
# --------------------------------------------------

print("\n" + "-" * 60)

print("\nTOP 20 RANKED LEGAL PROVISIONS:")

for rank, item in enumerate(
    ranked_provisions,
    start=1
):

    print(
        f"\n{rank}. {item['provision']}"
    )

    print(
        f"   Category: "
        f"{item['category']}"
    )

    print(
        f"   Relevance Score: "
        f"{item['relevance_score']}"
    )

    print(
        f"   Confidence: "
        f"{item['confidence']}"
    )

    print(
        f"   Pages: "
        f"{item['page_count']}"
    )

    print(
        f"   Evidence Records: "
        f"{item['evidence_count']}"
    )

    print(
        f"   Confidence Counts: "
        f"{item['confidence_counts']}"
    )

# --------------------------------------------------
# 9. Sample evidence
# --------------------------------------------------

print("\n" + "-" * 60)

print("\nSAMPLE EVIDENCE:")

sample_found = False

if provisions["articles"]:

    article = next(
        iter(provisions["articles"])
    )

    evidence = (
        provisions["articles"][article]
        ["evidence"][0]
    )

    print(f"\n{article}")
    print(
        f"Page: {evidence['page']}"
    )
    print(
        f"Evidence: "
        f"{evidence['text'][:500]}"
    )

    sample_found = True


if provisions["sections"]:

    section = next(
        iter(provisions["sections"])
    )

    evidence = (
        provisions["sections"][section]
        ["evidence"][0]
    )

    print(f"\n{section}")
    print(
        f"Page: {evidence['page']}"
    )
    print(
        f"Evidence: "
        f"{evidence['text'][:500]}"
    )

    sample_found = True


if provisions["constitutional_amendments"]:

    amendment = next(
        iter(
            provisions[
                "constitutional_amendments"
            ]
        )
    )

    evidence = (
        provisions[
            "constitutional_amendments"
        ][amendment]["evidence"][0]
    )

    print(f"\n{amendment}")
    print(
        f"Page: {evidence['page']}"
    )
    print(
        f"Evidence: "
        f"{evidence['text'][:500]}"
    )

    sample_found = True


if not sample_found:
    print("No evidence records found.")

# --------------------------------------------------
# 10. Extraction totals
# --------------------------------------------------

print("\n" + "-" * 60)

print(
    f"\nTotal Articles: "
    f"{len(provisions['articles'])}"
)

print(
    f"Total Sections: "
    f"{len(provisions['sections'])}"
)

print(
    f"Total Constitutional Amendments: "
    f"{len(provisions['constitutional_amendments'])}"
)

# --------------------------------------------------
# 11. JSON serialization test
# --------------------------------------------------

print("\nTesting JSON serialization...")

json_data = json.dumps(
    validated_provisions,
    indent=2,
    ensure_ascii=False
)

print(
    f"JSON size: "
    f"{len(json_data):,} characters"
)

print(
    "JSON serialization successful!"
)

# --------------------------------------------------
# 12. Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print(
    "LEGAL PROVISION EXTRACTION TEST "
    "COMPLETED"
)
print("=" * 60)
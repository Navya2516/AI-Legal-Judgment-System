import sys
from pathlib import Path
import json


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT SERVICES
# ============================================================

from backend.app.services.pdf_service import (
    extract_pdf_text
)

from backend.app.services.preprocessing_service import (
    preprocess_judgment
)

from backend.app.services.legal_provision_service import (
    extract_legal_provisions_with_evidence,
    validate_legal_provisions,
    rank_legal_provisions
)

from backend.app.services.case_continuation_service import (
    generate_case_continuation_insights
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "judgments"
    / "kesavananda_bharati_1973.pdf"
)

TOP_K = 5

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_insights"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "case_continuation_insights.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("CASE CONTINUATION INSIGHTS TEST")
print("=" * 60)


# ============================================================
# STEP 1
# ============================================================

print("\n1. Extracting PDF text...")

extracted_data = extract_pdf_text(
    str(PDF_PATH)
)

print(
    f"   Pages extracted: "
    f"{extracted_data['total_pages']}"
)


# ============================================================
# STEP 2
# ============================================================

print("\n2. Preprocessing judgment...")

processed_data = preprocess_judgment(
    extracted_data
)

print(
    f"   Pages processed: "
    f"{processed_data['total_pages']}"
)


# ============================================================
# STEP 3
# ============================================================

print("\n3. Extracting legal provisions...")

provisions = (
    extract_legal_provisions_with_evidence(
        processed_data["pages"]
    )
)

print(
    f"   Articles found: "
    f"{len(provisions['articles'])}"
)

print(
    f"   Sections found: "
    f"{len(provisions['sections'])}"
)

print(
    f"   Amendments found: "
    f"{len(provisions['constitutional_amendments'])}"
)


# ============================================================
# STEP 4
# ============================================================

print("\n4. Validating provisions...")

validated_provisions = (
    validate_legal_provisions(
        provisions
    )
)

print(
    "   Validation completed."
)


# ============================================================
# STEP 5
# ============================================================

print("\n5. Ranking provisions...")

ranked_provisions = (
    rank_legal_provisions(
        validated_provisions,
        top_k=TOP_K
    )
)

print(
    f"   Top {TOP_K} provisions selected."
)


# ============================================================
# STEP 6
# ============================================================

print("\n6. Selected provisions:")

for index, item in enumerate(
    ranked_provisions,
    start=1
):

    print(
        f"   {index}. "
        f"{item['provision']} "
        f"({item['relevance_score']})"
    )


# ============================================================
# STEP 7
# ============================================================

print(
    "\n7. Generating continuation insights..."
)

try:

    insights = (
        generate_case_continuation_insights(
            ranked_provisions,
            validated_provisions
        )
    )

    print(
        "   [PASS] Insights generated."
    )

except Exception as error:

    print(
        f"   [ERROR] {error}"
    )

    raise


# ============================================================
# STEP 8 — DISPLAY SUMMARY
# ============================================================

print("\n" + "-" * 60)

print(
    "\n8. Generated insight summary:"
)

print(
    "\nPotential legal provisions:"
)

for item in insights[
    "potential_legal_provisions"
]:

    print(
        f"   - {item.get('provision')}"
    )

    print(
        f"     Confidence: "
        f"{item.get('confidence')}"
    )

print(
    "\nIssues for continuation:"
)

for item in insights[
    "issues_for_continuation"
]:

    print(
        f"   - {item.get('issue')}"
    )

    print(
        f"     Confidence: "
        f"{item.get('confidence')}"
    )

print(
    "\nRelated provisions:"
)

for item in insights[
    "related_provisions"
]:

    print(
        f"   - {item.get('provision')}"
    )


# ============================================================
# STEP 9 — SAVE
# ============================================================

print(
    "\n9. Saving insights..."
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output = {
    "document": {
        "filename": PDF_PATH.name,
        "total_pages":
            processed_data[
                "total_pages"
            ]
    },

    "input": {
        "top_k": TOP_K,
        "selected_provisions": [
            item["provision"]
            for item in ranked_provisions
        ]
    },

    "insights": insights
}

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(
        output,
        output_file,
        indent=2,
        ensure_ascii=False
    )


print(
    f"   Saved to:"
)

print(
    f"   {OUTPUT_PATH}"
)


# ============================================================
# STEP 10 — VALIDATE
# ============================================================

print(
    "\n10. Validating final output..."
)

required_fields = [
    "potential_legal_provisions",
    "issues_for_continuation",
    "related_provisions",
    "overall_limitation",
    "disclaimer"
]

for field in required_fields:

    if field not in insights:

        raise ValueError(
            f"Missing output field: {field}"
        )

    print(
        f"   [PASS] {field}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print(
    "CASE CONTINUATION INSIGHTS TEST COMPLETED"
)

print("=" * 60)

print(
    f"\nOutput file:"
)

print(
    OUTPUT_PATH
)

print("=" * 60)
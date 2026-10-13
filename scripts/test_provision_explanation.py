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
    validate_legal_provisions
)

from backend.app.services.provision_explanation_service import (
    generate_provision_explanation
)


# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "judgments"
    / "kesavananda_bharati_1973.pdf"
)


# ============================================================
# TEST
# ============================================================

print("=" * 60)
print("PROVISION EXPLANATION TEST")
print("=" * 60)


# ============================================================
# STEP 1 — EXTRACT PDF
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
# STEP 2 — PREPROCESS
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
# STEP 3 — EXTRACT PROVISIONS
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
# STEP 4 — VALIDATE PROVISIONS
# ============================================================

print("\n4. Validating legal provisions...")

validated_provisions = (
    validate_legal_provisions(
        provisions
    )
)

print(
    "   Validation completed."
)


# ============================================================
# STEP 5 — SELECT ARTICLE 368
# ============================================================

print("\n5. Selecting Article 368...")

PROVISION = "Article 368"

if PROVISION not in (
    validated_provisions["articles"]
):

    raise ValueError(
        f"{PROVISION} was not found in the judgment."
    )

provision_data = (
    validated_provisions[
        "articles"
    ][PROVISION]
)

print(
    f"   Provision: {PROVISION}"
)

print(
    f"   Pages: "
    f"{len(provision_data['pages'])}"
)

print(
    f"   Evidence records: "
    f"{len(provision_data['evidence'])}"
)


# ============================================================
# STEP 6 — GENERATE EXPLANATION
# ============================================================

print(
    "\n6. Generating evidence-grounded explanation..."
)

result = generate_provision_explanation(
    PROVISION,
    provision_data
)


# ============================================================
# STEP 7 — DISPLAY RESULT
# ============================================================

print("\n" + "-" * 60)

print("\nGENERATED PROVISION EXPLANATION:")

print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )
)


# ============================================================
# STEP 8 — SAVE RESULT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "provision_explanations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "article_368_explanation.json"
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(
        result,
        output_file,
        indent=2,
        ensure_ascii=False
    )


print(
    f"\nSaved explanation to:"
)

print(
    f"   {OUTPUT_PATH}"
)


# ============================================================
# STEP 9 — BASIC VALIDATION
# ============================================================

print("\n" + "-" * 60)

required_fields = [
    "provision",
    "what_the_provision_is_discussed_as",
    "why_it_appears_in_the_judgment",
    "court_discussion",
    "evidence",
    "confidence",
    "limitation"
]

print(
    "\nVALIDATING GENERATED OUTPUT..."
)

for field in required_fields:

    if field in result:

        print(
            f"   [PASS] {field}"
        )

    else:

        print(
            f"   [FAIL] {field}"
        )

        raise ValueError(
            f"Missing required field: {field}"
        )


print("\n" + "=" * 60)
print(
    "PROVISION EXPLANATION TEST COMPLETED"
)
print("=" * 60)
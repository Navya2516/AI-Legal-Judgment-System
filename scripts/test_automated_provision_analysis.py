import sys
from pathlib import Path
import json
import time


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

from backend.app.services.provision_explanation_service import (
    generate_provision_explanation
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
    / "provision_explanations"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "legal_provision_analysis.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AUTOMATED PROVISION ANALYSIS TEST")
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
# STEP 4 — VALIDATE
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
# STEP 5 — RANK
# ============================================================

print("\n5. Ranking legal provisions...")

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
# DISPLAY SELECTED PROVISIONS
# ============================================================

print("\n" + "-" * 60)

print(
    f"\nSELECTED TOP {TOP_K} PROVISIONS:"
)

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


# ============================================================
# STEP 6 — GENERATE EXPLANATIONS
# ============================================================

print("\n" + "-" * 60)

print(
    "\n6. Generating explanations..."
)

analysis_results = []

for rank, ranked_item in enumerate(
    ranked_provisions,
    start=1
):

    provision = (
        ranked_item["provision"]
    )

    category = (
        ranked_item["category"]
    )

    print(
        f"\n[{rank}/{TOP_K}] "
        f"Processing {provision}..."
    )

    # --------------------------------------------------------
    # Locate original validated provision data
    # --------------------------------------------------------

    if category == "Article":

        provision_data = (
            validated_provisions[
                "articles"
            ].get(
                provision
            )
        )

    elif category == "Section":

        provision_data = (
            validated_provisions[
                "sections"
            ].get(
                provision
            )
        )

    elif category == (
        "Constitutional Amendment"
    ):

        provision_data = (
            validated_provisions[
                "constitutional_amendments"
            ].get(
                provision
            )
        )

    else:

        print(
            f"   [SKIP] Unknown category: "
            f"{category}"
        )

        continue

    # --------------------------------------------------------
    # Check provision data
    # --------------------------------------------------------

    if not provision_data:

        print(
            f"   [SKIP] No data found for "
            f"{provision}"
        )

        continue

    # --------------------------------------------------------
    # Generate explanation
    # --------------------------------------------------------

    try:

        explanation = (
            generate_provision_explanation(
                provision,
                provision_data
            )
        )

    except Exception as error:

        print(
            f"   [ERROR] Failed to generate "
            f"explanation: {error}"
        )

        continue

    # --------------------------------------------------------
    # Add ranking information
    # --------------------------------------------------------

    result = {
        "rank": rank,

        "provision": provision,

        "category": category,

        "relevance_score":
            ranked_item[
                "relevance_score"
            ],

        "ranking_confidence":
            ranked_item[
                "confidence"
            ],

        "page_count":
            ranked_item[
                "page_count"
            ],

        "evidence_count":
            ranked_item[
                "evidence_count"
            ],

        "explanation":
            explanation
    }

    analysis_results.append(
        result
    )

    print(
        f"   [PASS] Explanation generated "
        f"for {provision}"
    )

    # --------------------------------------------------------
    # Small delay between API requests
    # --------------------------------------------------------

    if rank < TOP_K:

        time.sleep(1)


# ============================================================
# STEP 7 — SAVE RESULTS
# ============================================================

print("\n" + "-" * 60)

print(
    "\n7. Saving automated analysis..."
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

final_output = {
    "document": {
        "filename": PDF_PATH.name,
        "total_pages":
            processed_data[
                "total_pages"
            ]
    },

    "analysis": {
        "top_k": TOP_K,
        "provisions_processed":
            len(analysis_results)
    },

    "provisions":
        analysis_results
}

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(
        final_output,
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
# STEP 8 — VALIDATE OUTPUT
# ============================================================

print("\n" + "-" * 60)

print(
    "\n8. Validating output..."
)

required_explanation_fields = [
    "provision",
    "what_the_provision_is_discussed_as",
    "why_it_appears_in_the_judgment",
    "court_discussion",
    "evidence",
    "confidence",
    "limitation"
]

for item in analysis_results:

    provision = item[
        "provision"
    ]

    explanation = item[
        "explanation"
    ]

    print(
        f"\n   Checking {provision}..."
    )

    for field in required_explanation_fields:

        if field not in explanation:

            raise ValueError(
                f"Missing field '{field}' "
                f"for {provision}"
            )

    print(
        f"   [PASS] {provision}"
    )


# ============================================================
# STEP 9 — FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "AUTOMATED PROVISION ANALYSIS COMPLETED"
)

print("=" * 60)

print(
    f"\nProvisions requested: {TOP_K}"
)

print(
    f"Provisions successfully processed: "
    f"{len(analysis_results)}"
)

print(
    f"\nOutput file:"
)

print(
    OUTPUT_PATH
)

print("=" * 60)
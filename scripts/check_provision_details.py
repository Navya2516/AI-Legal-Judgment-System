import sys
import json
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT SERVICE
# ============================================================

from backend.app.services.legal_provision_service import (
    extract_legal_provisions_with_evidence,
    validate_legal_provisions,
    rank_legal_provisions
)


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

STRUCTURED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploads"
    / DOCUMENT_ID
    / "structured.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LEGAL PROVISION EXTRACTION DIAGNOSTIC")
    print("=" * 70)

    if not STRUCTURED_FILE.exists():

        print("\nERROR:")
        print("Structured judgment file not found:")
        print(STRUCTURED_FILE)

        return

    # --------------------------------------------------------
    # Load judgment
    # --------------------------------------------------------

    with open(
        STRUCTURED_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        processed_data = json.load(file)

    pages = processed_data.get("pages", [])

    print(
        f"\nLoaded {len(pages)} judgment pages."
    )

    # --------------------------------------------------------
    # Extract provisions
    # --------------------------------------------------------

    print("\nExtracting provisions...")

    extracted = (
        extract_legal_provisions_with_evidence(
            pages
        )
    )

    print("\nRaw extraction completed.")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validated = (
        validate_legal_provisions(
            extracted
        )
    )

    print("Validation completed.")

    # ========================================================
    # ARTICLES
    # ========================================================

    articles = validated.get(
        "articles",
        {}
    )

    print("\n" + "=" * 70)
    print("ARTICLE DIAGNOSTIC")
    print("=" * 70)

    print(
        f"\nTotal article keys: {len(articles)}"
    )

    # Search Article 31 variants
    print("\nArticle 31 variants found:")

    found_31 = False

    for name, data in articles.items():

        name_lower = str(name).lower()

        if "31" in name_lower:

            found_31 = True

            print(
                f"\nProvision: {name}"
            )

            if isinstance(data, dict):

                print(
                    f"Pages: "
                    f"{data.get('pages', [])}"
                )

                print(
                    f"Confidence: "
                    f"{data.get('confidence')}"
                )

                evidence = data.get(
                    "evidence",
                    []
                )

                print(
                    f"Evidence count: "
                    f"{len(evidence)}"
                )

    if not found_31:

        print(
            "No Article 31-related key was found."
        )

    # ========================================================
    # ARTICLE 31A SEARCH IN RAW DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("SEARCHING RAW EXTRACTION FOR ARTICLE 31A")
    print("=" * 70)

    raw_articles = {}

    if isinstance(extracted, dict):

        raw_articles = extracted.get(
            "articles",
            {}
        )

    print(
        f"\nRaw article keys: "
        f"{len(raw_articles)}"
    )

    found_raw_31a = False

    for name, data in raw_articles.items():

        name_lower = str(name).lower()

        if (
            "31a" in name_lower
            or "31-a" in name_lower
            or "31 a" in name_lower
        ):

            found_raw_31a = True

            print(
                f"\nRAW MATCH: {name}"
            )

            if isinstance(data, dict):

                print(
                    f"Pages: "
                    f"{data.get('pages', [])}"
                )

                print(
                    f"Confidence: "
                    f"{data.get('confidence')}"
                )

    if not found_raw_31a:

        print(
            "\nArticle 31A was NOT found "
            "in raw extraction."
        )

    # ========================================================
    # CONSTITUTIONAL AMENDMENTS
    # ========================================================

    amendments = validated.get(
        "constitutional_amendments",
        {}
    )

    print("\n" + "=" * 70)
    print("CONSTITUTIONAL AMENDMENT DIAGNOSTIC")
    print("=" * 70)

    print(
        f"\nTotal amendment keys: "
        f"{len(amendments)}"
    )

    for name, data in amendments.items():

        print(
            f"\n{name}"
        )

        if isinstance(data, dict):

            print(
                f"Pages: "
                f"{data.get('pages', [])}"
            )

            print(
                f"Confidence: "
                f"{data.get('confidence')}"
            )

    # ========================================================
    # RANKING
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP RANKED PROVISIONS")
    print("=" * 70)

    try:

        ranked = rank_legal_provisions(
            validated
        )

        print(
            f"\nTotal ranked provisions: "
            f"{len(ranked)}"
        )

        for index, item in enumerate(
            ranked[:20],
            start=1
        ):

            print(
                f"\n{index}. {item}"
            )

    except Exception as exc:

        print(
            "\nRanking failed:"
        )

        print(
            str(exc)
        )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
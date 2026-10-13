import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRUCTURED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploads"
    / "0adde2b14334482d83ae091f61003bc6"
    / "structured.json"
)


def main():

    print("=" * 70)
    print("ARTICLE 31A TEXT DIAGNOSTIC")
    print("=" * 70)

    if not STRUCTURED_FILE.exists():
        print("\nERROR: structured.json not found.")
        print(STRUCTURED_FILE)
        return

    with open(
        STRUCTURED_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    pages = data.get("pages", [])

    patterns = [
        r"31A",
        r"31-A",
        r"31 A",
        r"31\s*[-–—]\s*A",
        r"Article\s+31",
    ]

    found_any = False

    for page in pages:

        text = page.get("text", "")

        # Look specifically for Article 31 variants
        matches = list(
            re.finditer(
                r"Article\s+31.{0,20}",
                text,
                flags=re.IGNORECASE
            )
        )

        if matches:

            found_any = True

            print(
                f"\n{'=' * 70}"
            )

            print(
                f"PAGE {page.get('page_number')}"
            )

            print(
                f"{'=' * 70}"
            )

            for match in matches:

                start = max(
                    0,
                    match.start() - 150
                )

                end = min(
                    len(text),
                    match.end() + 250
                )

                context = text[start:end]

                print("\n--- CONTEXT ---")
                print(context)

    if not found_any:

        print(
            "\nNo 'Article 31...' occurrences "
            "were found in structured text."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "SEARCHING FOR RAW 31-A / 31A VARIANTS"
    )

    print(
        "=" * 70
    )

    for page in pages:

        text = page.get("text", "")

        for pattern in patterns:

            matches = list(
                re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                )
            )

            if matches:

                print(
                    f"\nPage {page.get('page_number')}: "
                    f"{pattern} -> "
                    f"{len(matches)} match(es)"
                )

    print(
        "\n" + "=" * 70
    )

    print("DIAGNOSTIC COMPLETED")

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
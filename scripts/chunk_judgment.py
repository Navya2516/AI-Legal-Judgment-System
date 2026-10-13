import json
import re
import sys
from pathlib import Path


# Approximate target size for each chunk
TARGET_WORDS = 450

# Context overlap between chunks
OVERLAP_WORDS = 75


def clean_text(text: str) -> str:
    """Clean extracted judgment text."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Common encoding issues
    replacements = {
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬\x9d": '"',
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"'
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(text: str):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + TARGET_WORDS,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        )

        chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - OVERLAP_WORDS

    return chunks


def main():

    # ---------------------------------------------------------
    # Input argument
    # ---------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python scripts\\chunk_judgment.py "
            "<structured_json>"
        )

        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():

        print(
            f"Input file not found: {input_path}"
        )

        return

    # ---------------------------------------------------------
    # Dynamic output path
    # ---------------------------------------------------------

    output_path = input_path.with_name(
        input_path.stem.replace(
            "_structured",
            ""
        )
        + "_chunks.json"
    )

    print(
        "Loading structured judgment:"
    )

    print(input_path)

    # ---------------------------------------------------------
    # Load JSON
    # ---------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        judgment = json.load(file)

    all_chunks = []

    chunk_counter = 1

    # ---------------------------------------------------------
    # Process pages
    # ---------------------------------------------------------

    for page in judgment.get("pages", []):

        page_number = page.get(
            "page_number",
            1
        )

        page_text = clean_text(
            page.get("text", "")
        )

        if not page_text:
            continue

        page_chunks = split_into_chunks(
            page_text
        )

        for chunk_index, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            chunk = {

                "chunk_id":
                    f"chunk_{chunk_counter:05d}",

                "page_number":
                    page_number,

                "chunk_index_on_page":
                    chunk_index,

                "source_file":
                    judgment.get(
                        "source_file",
                        input_path.name
                    ),

                "text":
                    chunk_text,

                "word_count":
                    len(chunk_text.split())
            }

            all_chunks.append(chunk)

            chunk_counter += 1

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    result = {

        "source_file":
            judgment.get(
                "source_file",
                input_path.name
            ),

        "total_pages":
            judgment.get(
                "total_pages",
                1
            ),

        "total_chunks":
            len(all_chunks),

        "chunking_config": {

            "target_words":
                TARGET_WORDS,

            "overlap_words":
                OVERLAP_WORDS
        },

        "chunks":
            all_chunks
    }

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    word_counts = [
        chunk["word_count"]
        for chunk in all_chunks
    ]

    average_words = (
        sum(word_counts) / len(word_counts)
        if word_counts
        else 0
    )

    print()
    print("=" * 60)
    print("JUDGMENT CHUNKING COMPLETED")
    print("=" * 60)

    print(
        f"Pages processed : "
        f"{judgment.get('total_pages', 1)}"
    )

    print(
        f"Total chunks    : "
        f"{len(all_chunks)}"
    )

    print(
        f"Average words   : "
        f"{average_words:.1f}"
    )

    print(
        f"Target words    : "
        f"{TARGET_WORDS}"
    )

    print(
        f"Overlap words   : "
        f"{OVERLAP_WORDS}"
    )

    print(
        f"Output file     : "
        f"{output_path}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Preview
    # ---------------------------------------------------------

    print()
    print("FIRST 5 CHUNKS")
    print("-" * 60)

    for chunk in all_chunks[:5]:

        preview = chunk["text"][:250]

        print()
        print(
            chunk["chunk_id"]
        )

        print(
            f"Page: "
            f"{chunk['page_number']}"
        )

        print(
            f"Words: "
            f"{chunk['word_count']}"
        )

        print(
            f"Preview: "
            f"{preview}..."
        )


if __name__ == "__main__":
    main()
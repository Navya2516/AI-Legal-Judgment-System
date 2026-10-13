import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = Path("data/processed/embeddings")


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "python scripts/create_embeddings.py "
            "data/processed/<judgment>_chunks.json"
        )
        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    print(f"Loading chunks:")
    print(input_path)

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    chunks = data["chunks"]

    if not chunks:
        print("No chunks found.")
        return

    print(f"Chunks loaded: {len(chunks)}")

    # Use the judgment filename as the output prefix
    # Example:
    # ravindra_kumar_2024_insc_131_chunks.json
    # -> ravindra_kumar_2024_insc_131
    prefix = input_path.stem.replace("_chunks", "")

    embeddings_path = (
        OUTPUT_DIR / f"{prefix}_embeddings.npy"
    )

    metadata_path = (
        OUTPUT_DIR / f"{prefix}_metadata.json"
    )

    print()
    print("Loading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print()
    print("Creating embeddings...")
    print("This may take a few minutes.")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        embeddings_path,
        embeddings
    )

    metadata = []

    for chunk in chunks:

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "chunk_index_on_page": chunk[
                "chunk_index_on_page"
            ],
            "source_file": chunk["source_file"],
            "text": chunk["text"],
            "word_count": chunk["word_count"]
        })

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("EMBEDDING CREATION COMPLETED")
    print("=" * 60)

    print(
        f"Judgment         : {prefix}"
    )

    print(
        f"Chunks embedded  : {len(chunks)}"
    )

    print(
        f"Embedding shape   : {embeddings.shape}"
    )

    print(
        f"Embedding size    : {embeddings.shape[1]}"
    )

    print(
        f"Embeddings saved  : {embeddings_path}"
    )

    print(
        f"Metadata saved    : {metadata_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
import json
import sys
from pathlib import Path

import faiss
import numpy as np


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "python scripts/build_faiss_index.py "
            "data/processed/embeddings/<name>_embeddings.npy"
        )
        return

    embeddings_path = Path(sys.argv[1])

    if not embeddings_path.exists():
        print(f"Embeddings not found: {embeddings_path}")
        return

    # Example:
    # ravindra_kumar_2024_insc_131_embeddings.npy
    # -> ravindra_kumar_2024_insc_131
    prefix = embeddings_path.stem.replace("_embeddings", "")

    output_dir = embeddings_path.parent

    metadata_path = (
        output_dir / f"{prefix}_metadata.json"
    )

    index_path = (
        output_dir / f"{prefix}.index"
    )

    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return

    print("Loading embeddings...")
    print(embeddings_path)

    embeddings = np.load(
        embeddings_path
    ).astype("float32")

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    # Embeddings were normalized during creation,
    # so Inner Product = cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    print()
    print("Building FAISS index...")

    index.add(
        embeddings
    )

    print(
        f"Vectors added: {index.ntotal}"
    )

    # Verify metadata count
    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if len(metadata) != index.ntotal:

        print()
        print("ERROR:")
        print(
            "Number of metadata records does not "
            "match number of vectors."
        )

        print(
            f"Metadata: {len(metadata)}"
        )

        print(
            f"Vectors: {index.ntotal}"
        )

        return

    faiss.write_index(
        index,
        str(index_path)
    )

    print()
    print("=" * 60)
    print("FAISS INDEX CREATION COMPLETED")
    print("=" * 60)

    print(
        f"Vectors indexed  : {index.ntotal}"
    )

    print(
        f"Vector dimension  : {dimension}"
    )

    print(
        "Index type        : IndexFlatIP"
    )

    print(
        f"Index saved       : {index_path}"
    )

    print(
        f"Metadata used     : {metadata_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
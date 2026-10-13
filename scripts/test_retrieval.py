import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_DIR = Path(
    "data/processed/embeddings"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def load_resources(judgment_name):

    index_path = (
        EMBEDDINGS_DIR /
        f"{judgment_name}.index"
    )

    metadata_path = (
        EMBEDDINGS_DIR /
        f"{judgment_name}_metadata.json"
    )

    if not index_path.exists():
        print(f"FAISS index not found: {index_path}")
        return None

    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return None

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(index_path)
    )

    print(
        f"Vectors in index: {index.ntotal}"
    )

    print("Loading metadata...")

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"Metadata records: {len(metadata)}"
    )

    if len(metadata) != index.ntotal:
        print(
            "ERROR: Metadata count does not "
            "match FAISS index."
        )
        return None

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    return index, metadata, model


def retrieve(
    question,
    index,
    metadata,
    model,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        chunk = metadata[index_id]

        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"]
        })

    return results


def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python scripts\\test_retrieval.py "
            "<judgment_name>"
        )

        print()
        print("Example:")

        print(
            "python scripts\\test_retrieval.py "
            "ravindra_kumar_2024_insc_131"
        )

        return

    judgment_name = sys.argv[1]

    print()
    print("=" * 60)
    print("LEGAL JUDGMENT RETRIEVAL")
    print("=" * 60)

    print(
        f"Judgment: {judgment_name}"
    )

    print("=" * 60)
    print()

    resources = load_resources(
        judgment_name
    )

    if resources is None:
        return

    index, metadata, model = resources

    while True:

        question = input(
            "\nEnter your question (or 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            continue

        results = retrieve(
            question,
            index,
            metadata,
            model
        )

        print()
        print("=" * 70)
        print("RETRIEVED CONTEXT")
        print("=" * 70)

        for rank, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"Rank       : {rank}"
            )

            print(
                f"Score      : "
                f"{result['score']:.4f}"
            )

            print(
                f"Chunk      : "
                f"{result['chunk_id']}"
            )

            print(
                f"Page       : "
                f"{result['page_number']}"
            )

            print("-" * 70)

            print(
                result["text"]
            )

            print("-" * 70)


if __name__ == "__main__":
    main()
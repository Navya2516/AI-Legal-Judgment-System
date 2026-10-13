import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "processed" / "embeddings"


class RetrievalService:

    def __init__(self, judgment_name):
        self.judgment_name = judgment_name

        self.index_path = (
            EMBEDDINGS_DIR / f"{judgment_name}.index"
        )

        self.metadata_path = (
            EMBEDDINGS_DIR / f"{judgment_name}_metadata.json"
        )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found:\n{self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found:\n{self.metadata_path}"
            )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print(f"Vectors in index: {self.index.ntotal}")

        print("Loading metadata...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Metadata records: {len(self.metadata)}")

        print("Loading embedding model...")
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def retrieve(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            if idx >= len(self.metadata):
                continue

            chunk = self.metadata[idx]

            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "text": chunk.get("text", ""),
                "metadata": chunk
            })

        return results


if __name__ == "__main__":

    service = RetrievalService(
        "ravindra_kumar_2024_insc_131"
    )

    query = input("\nEnter your legal query: ")

    results = service.retrieve(
        query,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for result in results:

        print(f"\nRank  : {result['rank']}")
        print(f"Score : {result['score']:.4f}")
        print(f"Chunk : {result['chunk_id']}")
        print(f"Page  : {result['page']}")
        print("-" * 70)

        print(result["text"])
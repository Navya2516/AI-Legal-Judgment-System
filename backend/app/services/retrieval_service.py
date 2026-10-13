import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_PATH = Path("data/processed/embeddings")


class RetrievalService:

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path
    ):

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(self.index_path)
        )

        print(
            f"Vectors in index: {self.index.ntotal}"
        )

        print("Loading metadata...")

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        print(
            f"Metadata records: {len(self.metadata)}"
        )

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "Metadata count does not match FAISS index."
            )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5
    ):

        if not question.strip():
            return []

        top_k = min(
            top_k,
            self.index.ntotal
        )

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id < 0:
                continue

            chunk = self.metadata[index_id]

            results.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"]
            })

        return results
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_WORDS = 450
OVERLAP_WORDS = 75


# ============================================================
# CHUNKING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted judgment text.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Fix common PDF encoding issues.
    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"')
    text = text.replace("â€\x9d", '"')

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(text: str) -> list:
    """
    Split page text into overlapping word-based chunks.
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

        chunk_words = words[start:end]

        chunk_text = " ".join(
            chunk_words
        )

        chunks.append(
            chunk_text
        )

        if end >= len(words):
            break

        start = end - OVERLAP_WORDS

    return chunks


def create_chunks(judgment: dict) -> dict:
    """
    Create chunks dynamically from a processed judgment.
    """

    all_chunks = []

    chunk_counter = 1

    for page in judgment.get("pages", []):

        page_number = page.get(
            "page_number"
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
                "chunk_id": (
                    f"chunk_{chunk_counter:05d}"
                ),
                "page_number": page_number,
                "chunk_index_on_page": chunk_index,
                "source_file": judgment.get(
                    "source_file",
                    "unknown"
                ),
                "text": chunk_text,
                "word_count": len(
                    chunk_text.split()
                )
            }

            all_chunks.append(
                chunk
            )

            chunk_counter += 1

    return {
        "source_file": judgment.get(
            "source_file",
            "unknown"
        ),
        "total_pages": judgment.get(
            "total_pages",
            len(judgment.get("pages", []))
        ),
        "total_chunks": len(
            all_chunks
        ),
        "chunking_config": {
            "target_words": TARGET_WORDS,
            "overlap_words": OVERLAP_WORDS
        },
        "chunks": all_chunks
    }


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(chunks: list):
    """
    Create normalized sentence embeddings for chunks.
    """

    if not chunks:
        raise ValueError(
            "No chunks available for embedding."
        )

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# ============================================================
# METADATA
# ============================================================

def create_metadata(chunks: list) -> list:
    """
    Create retrieval metadata corresponding to each vector.
    """

    metadata = []

    for chunk in chunks:

        metadata.append({
            "chunk_id": chunk[
                "chunk_id"
            ],
            "page_number": chunk[
                "page_number"
            ],
            "chunk_index_on_page": chunk[
                "chunk_index_on_page"
            ],
            "source_file": chunk[
                "source_file"
            ],
            "text": chunk[
                "text"
            ],
            "word_count": chunk[
                "word_count"
            ]
        })

    return metadata


# ============================================================
# FAISS INDEX
# ============================================================

def create_faiss_index(
    embeddings: np.ndarray
):
    """
    Create a FAISS inner-product index.

    Because embeddings are normalized, inner product
    corresponds to cosine similarity.
    """

    if len(embeddings) == 0:
        raise ValueError(
            "No embeddings available."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# BUILD DOCUMENT INDEX
# ============================================================

def build_document_index(
    structured_file: Path,
    document_id: str
) -> dict:
    """
    Build a complete searchable index for one uploaded judgment.

    Output:
        data/processed/documents/{document_id}/
    """

    if not structured_file.exists():
        raise FileNotFoundError(
            f"Structured judgment not found: {structured_file}"
        )

    # --------------------------------------------------------
    # Load processed judgment
    # --------------------------------------------------------

    with open(
        structured_file,
        "r",
        encoding="utf-8"
    ) as file:

        judgment = json.load(file)

    # --------------------------------------------------------
    # Create document directory
    # --------------------------------------------------------

    output_dir = (
        Path("data/processed/documents")
        / document_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Chunk judgment
    # --------------------------------------------------------

    chunk_data = create_chunks(
        judgment
    )

    chunks = chunk_data["chunks"]

    if not chunks:
        raise ValueError(
            "No usable text chunks were created."
        )

    chunks_file = (
        output_dir / "chunks.json"
    )

    with open(
        chunks_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunk_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        chunks
    )

    embeddings_file = (
        output_dir / "embeddings.npy"
    )

    np.save(
        embeddings_file,
        embeddings
    )

    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    metadata = create_metadata(
        chunks
    )

    metadata_file = (
        output_dir / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    index = create_faiss_index(
        embeddings
    )

    index_file = (
        output_dir
        / "legal_judgment.index"
    )

    faiss.write_index(
        index,
        str(index_file)
    )

    # --------------------------------------------------------
    # Return information
    # --------------------------------------------------------

    return {
        "document_id": document_id,
        "total_pages": chunk_data[
            "total_pages"
        ],
        "total_chunks": len(
            chunks
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "index_vectors": int(
            index.ntotal
        ),
        "chunks_path": str(
            chunks_file
        ),
        "embeddings_path": str(
            embeddings_file
        ),
        "metadata_path": str(
            metadata_file
        ),
        "index_path": str(
            index_file
        )
    }
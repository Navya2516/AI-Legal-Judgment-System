import re
from typing import Any, Dict, List


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ============================================================
# SPLIT ANSWER INTO CLAIMS
# ============================================================

def split_into_claims(
    answer: str
) -> List[str]:
    """
    Split an LLM answer into simple candidate claims.

    This is intentionally lightweight. We are not trying
    to perform full semantic parsing here.
    """

    if not answer:
        return []

    text = answer.strip()

    # Remove markdown bullets / numbering.
    text = re.sub(
        r"(?m)^\s*[-*]\s+",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*\d+\.\s+",
        "",
        text
    )

    # Split on sentence boundaries.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for sentence in sentences:

        sentence = normalize_text(
            sentence
        )

        if not sentence:
            continue

        if len(sentence) < 10:
            continue

        claims.append(
            sentence
        )

    return claims


# ============================================================
# EVIDENCE SENTENCE MATCHING
# ============================================================

def tokenize(text: str) -> set:
    """
    Create a simple set of normalized words.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    # Remove extremely common words.
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "and",
        "in",
        "on",
        "for",
        "that",
        "this",
        "it",
        "as",
        "by",
        "with",
        "from",
        "under",
    }

    return {
        word
        for word in words
        if word not in stopwords
    }


def calculate_word_overlap(
    claim: str,
    evidence: str
) -> float:

    claim_words = tokenize(
        claim
    )

    evidence_words = tokenize(
        evidence
    )

    if not claim_words:
        return 0.0

    overlap = (
        claim_words
        & evidence_words
    )

    return len(overlap) / len(
        claim_words
    )


# ============================================================
# VALIDATE CLAIMS
# ============================================================

def validate_claims(
    answer: str,
    evidence_chunks: List[Dict[str, Any]],
    minimum_overlap: float = 0.35
) -> Dict[str, Any]:
    """
    Lightweight claim-level validation.

    Each sentence is compared against the retrieved evidence.
    The strongest matching evidence chunk is recorded.

    This is a heuristic, not a legal truth detector.
    """

    claims = split_into_claims(
        answer
    )

    results = []

    for claim in claims:

        best_score = 0.0
        best_page = None
        best_chunk_id = None

        for chunk in evidence_chunks:

            evidence_text = normalize_text(
                chunk.get("text", "")
            )

            score = calculate_word_overlap(
                claim,
                evidence_text
            )

            if score > best_score:

                best_score = score

                best_page = chunk.get(
                    "page_number"
                )

                best_chunk_id = chunk.get(
                    "chunk_id"
                )

        results.append({

            "claim": claim,

            "supported": (
                best_score >= minimum_overlap
            ),

            "overlap_score": round(
                best_score,
                3
            ),

            "evidence_page": best_page,

            "evidence_chunk": best_chunk_id,
        })

    unsupported = [
        item
        for item in results
        if not item["supported"]
    ]

    return {

        "total_claims": len(
            results
        ),

        "supported_claims": (
            len(results)
            - len(unsupported)
        ),

        "unsupported_claims": len(
            unsupported
        ),

        "faithfulness_score": (
            (
                len(results)
                - len(unsupported)
            )
            / len(results)
            if results
            else 0.0
        ),

        "claims": results,
    }
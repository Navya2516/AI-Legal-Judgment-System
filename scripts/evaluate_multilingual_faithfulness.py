import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_summaries"
    / f"{DOCUMENT_ID}_summary.json"
)

MULTILINGUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multilingual_explanations"
    / f"{DOCUMENT_ID}_tamil.json"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "multilingual_faithfulness_results.json"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Initial baseline threshold.
# This will be calibrated later using manually verified examples.
FAITHFULNESS_THRESHOLD = 0.60


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_page_citations(text):
    """
    Extract citations such as:

    [Page 128]
    [Page 236]
    """

    return [
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            text,
            flags=re.IGNORECASE
        )
    ]


def remove_citations(text):
    """
    Remove page citations before semantic comparison.
    """

    text = re.sub(
        r"\[Page\s+\d+\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def split_claims(text):
    """
    Split explanation into sentence-level claims.

    Handles:
    - English punctuation
    - Tamil punctuation
    - Devanagari danda
    """

    text = text.replace("\n", " ")

    sentences = re.split(
        r"(?<=[.!?\u0964\u0965\u0be4])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 20
    ]


def encode_texts(model, texts):
    """
    Generate normalized multilingual embeddings.
    """

    return model.encode(
        texts,
        normalize_embeddings=True
    )


def calculate_similarity(
    model,
    source_text,
    target_text
):
    """
    Calculate cross-language semantic similarity.

    Example:

    English source
          ↕
    Tamil explanation
    """

    source_text = remove_citations(
        source_text
    )

    target_text = remove_citations(
        target_text
    )

    if not source_text or not target_text:
        return 0.0

    embeddings = encode_texts(
        model,
        [
            source_text,
            target_text
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


def build_source_claims(summary):
    """
    Build the English source claims that the Tamil
    simplified explanation should preserve.

    These come directly from the generated English
    judgment summary.
    """

    source_simplified = summary.get(
        "simplified_explanation",
        ""
    )

    if not source_simplified:
        raise ValueError(
            "English simplified_explanation not found."
        )

    return split_claims(
        source_simplified
    )


def build_supporting_context(summary):
    """
    Build additional English source context.

    This is NOT used as the primary translation
    comparison. It is retained as supporting context
    for interpretation/reporting.
    """

    return {
        "case_overview": summary.get(
            "case_overview",
            ""
        ),

        "court_reasoning": summary.get(
            "court_reasoning",
            ""
        ),

        "decision_or_conclusion": summary.get(
            "decision_or_conclusion",
            ""
        )
    }


def main():

    print("=" * 70)
    print(
        "MULTILINGUAL CROSS-LANGUAGE "
        "FAITHFULNESS EVALUATION"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load source summary and Tamil explanation
    # ---------------------------------------------------------

    summary_data = load_json(
        SUMMARY_FILE
    )

    multilingual_data = load_json(
        MULTILINGUAL_FILE
    )

    summary = summary_data["summary"]

    explanation = multilingual_data[
        "explanation"
    ]

    english_simplified = summary.get(
        "simplified_explanation",
        ""
    )

    tamil_simplified = explanation.get(
        "simplified_explanation",
        ""
    )

    if not english_simplified:
        raise ValueError(
            "English simplified explanation not found."
        )

    if not tamil_simplified:
        raise ValueError(
            "Tamil simplified explanation not found."
        )

    print(
        "\nEnglish source summary loaded."
    )

    print(
        "Tamil explanation loaded."
    )

    # ---------------------------------------------------------
    # 2. Load multilingual model
    # ---------------------------------------------------------

    print(
        "\nLoading multilingual embedding model..."
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Multilingual model loaded successfully."
    )

    # ---------------------------------------------------------
    # 3. Extract claims
    # ---------------------------------------------------------

    english_claims = build_source_claims(
        summary
    )

    tamil_claims = split_claims(
        tamil_simplified
    )

    print(
        f"\nEnglish claims found : "
        f"{len(english_claims)}"
    )

    print(
        f"Tamil claims found   : "
        f"{len(tamil_claims)}"
    )

    # ---------------------------------------------------------
    # 4. Citation evaluation
    # ---------------------------------------------------------

    tamil_citations = extract_page_citations(
        json.dumps(
            explanation,
            ensure_ascii=False
        )
    )

    unique_citations = sorted(
        set(tamil_citations)
    )

    total_pages = summary_data.get(
        "total_pages"
    )

    if not total_pages:

        # Fallback to the structured judgment
        structured_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "uploads"
            / DOCUMENT_ID
            / "structured.json"
        )

        if structured_file.exists():

            structured_data = load_json(
                structured_file
            )

            total_pages = len(
                structured_data.get(
                    "pages",
                    []
                )
            )

    if not total_pages:
        total_pages = 0

    valid_citations = [
        page
        for page in unique_citations
        if 1 <= page <= total_pages
    ]

    invalid_citations = [
        page
        for page in unique_citations
        if page < 1 or page > total_pages
    ]

    citation_validity = (
        len(valid_citations)
        / len(unique_citations)
        if unique_citations
        else 0
    )

    # ---------------------------------------------------------
    # 5. Cross-language claim comparison
    # ---------------------------------------------------------

    claim_results = []

    # Compare every English claim with every Tamil claim.
    # This avoids assuming that translation preserves sentence order.
    similarity_matrix = []

    for english_claim in english_claims:
        row = []
        for tamil_claim in tamil_claims:
            row.append(
                calculate_similarity(
                    model,
                    english_claim,
                    tamil_claim
                )
            )
        similarity_matrix.append(row)

    faithful_claims = 0
    similarities = []
    used_tamil = set()

    # Greedy one-to-one matching from strongest semantic matches first.
    candidate_pairs = []
    for english_index, row in enumerate(similarity_matrix):
        for tamil_index, similarity in enumerate(row):
            candidate_pairs.append(
                (similarity, english_index, tamil_index)
            )

    candidate_pairs.sort(reverse=True)

    matched_pairs = []
    used_english = set()

    for similarity, english_index, tamil_index in candidate_pairs:
        if english_index in used_english or tamil_index in used_tamil:
            continue

        used_english.add(english_index)
        used_tamil.add(tamil_index)
        matched_pairs.append(
            (english_index, tamil_index, similarity)
        )

    matched_pairs.sort(key=lambda item: item[0])

    for match_number, (english_index, tamil_index, similarity) in enumerate(
        matched_pairs,
        start=1
    ):

        english_claim = english_claims[english_index]
        tamil_claim = tamil_claims[tamil_index]

        similarities.append(similarity)

        faithful = (
            similarity
            >= FAITHFULNESS_THRESHOLD
        )

        if faithful:
            faithful_claims += 1

        claim_results.append({

            "claim_number":
                match_number,

            "english_claim_index":
                english_index + 1,

            "tamil_claim_index":
                tamil_index + 1,

            "english_source_claim":
                english_claim,

            "tamil_claim":
                tamil_claim,

            "semantic_similarity":
                round(
                    similarity,
                    4
                ),

            "faithfulness_threshold":
                FAITHFULNESS_THRESHOLD,

            "faithful":
                faithful
        })

    matched_claim_count = len(matched_pairs)

    # ---------------------------------------------------------
    # 6. Handle mismatched claim counts
    # ---------------------------------------------------------

    matched_english_indices = {
        english_index
        for english_index, _, _ in matched_pairs
    }

    matched_tamil_indices = {
        tamil_index
        for _, tamil_index, _ in matched_pairs
    }

    unmatched_english_claims = [
        claim
        for index, claim in enumerate(english_claims)
        if index not in matched_english_indices
    ]

    unmatched_tamil_claims = [
        claim
        for index, claim in enumerate(tamil_claims)
        if index not in matched_tamil_indices
    ]

    # ---------------------------------------------------------
    # 7. Faithfulness metrics
    # ---------------------------------------------------------

    total_compared_claims = len(
        claim_results
    )

    semantic_faithfulness_rate = (
        faithful_claims
        / total_compared_claims
        if total_compared_claims
        else 0
    )

    mean_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0
    )

    # ---------------------------------------------------------
    # 8. Citation coverage
    # ---------------------------------------------------------

    tamil_claims_with_citations = 0

    for claim in tamil_claims:

        if extract_page_citations(
            claim
        ):

            tamil_claims_with_citations += 1

    citation_coverage = (
        tamil_claims_with_citations
        / len(tamil_claims)
        if tamil_claims
        else 0
    )

    # ---------------------------------------------------------
    # 9. Supporting source context
    # ---------------------------------------------------------

    supporting_context = (
        build_supporting_context(
            summary
        )
    )

    # ---------------------------------------------------------
    # 10. Save results
    # ---------------------------------------------------------

    results = {

        "document_id":
            DOCUMENT_ID,

        "language":
            "Tamil",

        "evaluation_type":
            "cross_language_semantic_faithfulness",

        "embedding_model":
            MODEL_NAME,

        "faithfulness_threshold":
            FAITHFULNESS_THRESHOLD,

        "claim_matching_method":
            "one_to_one_best_semantic_match",

        "english_claim_count":
            len(english_claims),

        "tamil_claim_count":
            len(tamil_claims),

        "compared_claim_count":
            total_compared_claims,

        "faithful_claims":
            faithful_claims,

        "semantic_faithfulness_rate":
            round(
                semantic_faithfulness_rate,
                4
            ),

        "mean_cross_language_similarity":
            round(
                mean_similarity,
                4
            ),

        "total_pages":
            total_pages,

        "unique_citations":
            unique_citations,

        "valid_citations":
            valid_citations,

        "invalid_citations":
            invalid_citations,

        "citation_validity":
            round(
                citation_validity,
                4
            ),

        "tamil_claims_with_citations":
            tamil_claims_with_citations,

        "citation_coverage":
            round(
                citation_coverage,
                4
            ),

        "unmatched_english_claims":
            unmatched_english_claims,

        "unmatched_tamil_claims":
            unmatched_tamil_claims,

        "claim_results":
            claim_results,

        "supporting_source_context":
            supporting_context
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # 11. Console output
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nEnglish claims            : "
        f"{len(english_claims)}"
    )

    print(
        f"Tamil claims              : "
        f"{len(tamil_claims)}"
    )

    print(
        f"Compared claims           : "
        f"{total_compared_claims}"
    )

    print(
        f"Faithful claims           : "
        f"{faithful_claims}"
    )

    print(
        f"\nSemantic Faithfulness     : "
        f"{semantic_faithfulness_rate * 100:.2f}%"
    )

    print(
        f"Mean Cross-language       : "
        f"{mean_similarity:.4f}"
    )

    print(
        f"\nCitation Validity         : "
        f"{citation_validity * 100:.2f}%"
    )

    print(
        f"Citation Coverage        : "
        f"{citation_coverage * 100:.2f}%"
    )

    print(
        f"\nFaithfulness Threshold   : "
        f"{FAITHFULNESS_THRESHOLD:.2f}"
    )

    # ---------------------------------------------------------
    # 12. Claim-level results
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "CLAIM-LEVEL CROSS-LANGUAGE FAITHFULNESS"
    )
    print("-" * 70)

    for result in claim_results:

        print(
            f"\nClaim {result['claim_number']}:"
        )

        print(
            f"  Similarity : "
            f"{result['semantic_similarity']:.4f}"
        )

        print(
            f"  Faithful   : "
            f"{result['faithful']}"
        )

    # ---------------------------------------------------------
    # 13. Warnings
    # ---------------------------------------------------------

    if invalid_citations:

        print(
            "\nWARNING: Invalid citations:"
        )

        for page in invalid_citations:

            print(
                f"  [Page {page}]"
            )

    if unmatched_english_claims:

        print(
            "\nWARNING: Some English source claims "
            "had no corresponding Tamil claim."
        )

    if unmatched_tamil_claims:

        print(
            "\nWARNING: Some Tamil claims "
            "had no corresponding English claim."
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULT_FILE
    )


if __name__ == "__main__":
    main()
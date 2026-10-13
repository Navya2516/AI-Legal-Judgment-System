import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_ID = "0adde2b14334482d83ae091f61003bc6"

MULTILINGUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multilingual_explanations"
    / f"{DOCUMENT_ID}_tamil.json"
)

STRUCTURED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploads"
    / DOCUMENT_ID
    / "structured.json"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "multilingual_evaluation_results.json"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Initial evaluation threshold.
# We will calibrate this later using manually verified examples.
GROUNDING_THRESHOLD = 0.60

# Passage configuration
PASSAGE_WORDS = 120
PASSAGE_OVERLAP = 30


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_page_citations(text):
    """
    Extract citations such as:

    [Page 128]
    [Page 128] [Page 331]
    """

    return [
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            text,
            flags=re.IGNORECASE
        )
    ]


def get_page_text(structured_data, page_number):
    """
    Retrieve original judgment text for a page.
    """

    for page in structured_data.get("pages", []):

        if page.get("page_number") == page_number:
            return page.get("text", "")

    return ""


def split_claims(text):
    """
    Split Tamil explanation into sentence-level claims.

    Handles:
    - English punctuation
    - Tamil danda
    - Tamil full stop
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


def remove_citations(text):
    """
    Remove [Page X] markers before semantic comparison.
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


def split_into_passages(
    text,
    passage_words=PASSAGE_WORDS,
    overlap=PASSAGE_OVERLAP
):
    """
    Split a judgment page into overlapping word-based passages.

    Example:

    Passage 1 = words 1-120
    Passage 2 = words 91-210
    Passage 3 = words 181-300
    """

    text = remove_citations(text)

    words = text.split()

    if not words:
        return []

    if len(words) <= passage_words:
        return [text]

    step = passage_words - overlap

    passages = []

    start = 0

    while start < len(words):

        end = min(
            start + passage_words,
            len(words)
        )

        passage = " ".join(
            words[start:end]
        ).strip()

        if passage:
            passages.append(passage)

        if end >= len(words):
            break

        start += step

    return passages


def semantic_similarity(
    model,
    claim,
    evidence
):
    """
    Calculate semantic similarity between:

    Tamil claim
            and
    English evidence passage.

    The multilingual embedding model maps both
    languages into a shared vector space.
    """

    claim = remove_citations(claim)
    evidence = remove_citations(evidence)

    if not claim or not evidence:
        return 0.0

    embeddings = model.encode(
        [
            claim,
            evidence
        ],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


def build_cited_evidence(
    model,
    claim,
    citations,
    structured_data
):
    """
    Build passage-level evidence from all cited pages.

    Returns the strongest semantic match across
    all passages from all cited pages.
    """

    best_similarity = -1.0
    best_page = None
    best_passage = None

    valid_claim_citations = []

    evidence_records = []

    for page_number in citations:

        page_text = get_page_text(
            structured_data,
            page_number
        )

        if not page_text:
            continue

        valid_claim_citations.append(
            page_number
        )

        passages = split_into_passages(
            page_text
        )

        for passage_index, passage in enumerate(
            passages,
            start=1
        ):

            similarity = semantic_similarity(
                model,
                claim,
                passage
            )

            evidence_records.append({
                "page": page_number,
                "passage_index": passage_index,
                "similarity": similarity,
                "text": passage
            })

            if similarity > best_similarity:

                best_similarity = similarity
                best_page = page_number
                best_passage = passage

    similarities = sorted(
        (
            record["similarity"]
            for record in evidence_records
        ),
        reverse=True
    )

    top_k = min(3, len(similarities))

    return {
        "best_similarity": (
            best_similarity
            if best_similarity >= 0
            else 0.0
        ),
        "top_k_mean_similarity": (
            sum(similarities[:top_k]) / top_k
            if top_k
            else 0.0
        ),
        "best_page": best_page,
        "best_passage": best_passage,
        "valid_citations": valid_claim_citations,
        "evidence_records": evidence_records
    }


def main():

    print("=" * 70)
    print(
        "MULTILINGUAL TAMIL "
        "PASSAGE-LEVEL SEMANTIC GROUNDING"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load files
    # ---------------------------------------------------------

    multilingual_data = load_json(
        MULTILINGUAL_FILE
    )

    structured_data = load_json(
        STRUCTURED_FILE
    )

    explanation = multilingual_data["explanation"]

    simplified = explanation.get(
        "simplified_explanation",
        ""
    )

    if not simplified:

        raise ValueError(
            "simplified_explanation not found."
        )

    print("\nTamil explanation loaded.")
    print("Original judgment loaded.")

    # ---------------------------------------------------------
    # 2. Load multilingual embedding model
    # ---------------------------------------------------------

    print("\nLoading multilingual embedding model...")

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
    # 3. Citation evaluation
    # ---------------------------------------------------------

    all_citations = extract_page_citations(
        json.dumps(
            explanation,
            ensure_ascii=False
        )
    )

    unique_citations = sorted(
        set(all_citations)
    )

    total_pages = len(
        structured_data.get(
            "pages",
            []
        )
    )

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
    # 4. Claim extraction
    # ---------------------------------------------------------

    claims = split_claims(
        simplified
    )

    total_claims = len(
        claims
    )

    claims_with_citations = 0
    grounded_claims = 0

    claim_results = []

    # ---------------------------------------------------------
    # 5. Passage-level grounding
    # ---------------------------------------------------------

    for claim_number, claim in enumerate(
        claims,
        start=1
    ):

        citations = extract_page_citations(
            claim
        )

        if citations:

            claims_with_citations += 1

        evidence_result = build_cited_evidence(
            model=model,
            claim=claim,
            citations=citations,
            structured_data=structured_data
        )

        best_similarity = (
            evidence_result[
                "best_similarity"
            ]
        )

        grounded = (
            bool(
                evidence_result[
                    "valid_citations"
                ]
            )
            and
            best_similarity >=
            GROUNDING_THRESHOLD
        )

        if grounded:

            grounded_claims += 1

        claim_results.append({

            "claim_number": claim_number,

            "claim": claim,

            "citations": citations,

            "valid_citations": (
                evidence_result[
                    "valid_citations"
                ]
            ),

            "best_evidence_page": (
                evidence_result[
                    "best_page"
                ]
            ),

            "best_passage_similarity": round(
                best_similarity,
                4
            ),

            "grounding_threshold": (
                GROUNDING_THRESHOLD
            ),

            "grounded": grounded,

            "best_evidence_passage": (
                evidence_result[
                    "best_passage"
                ]
            )
        })

    # ---------------------------------------------------------
    # 6. Calculate metrics
    # ---------------------------------------------------------

    citation_coverage = (
        claims_with_citations
        / total_claims
        if total_claims
        else 0
    )

    semantic_grounding_rate = (
        grounded_claims
        / total_claims
        if total_claims
        else 0
    )

    cited_claim_grounding_rate = (
        grounded_claims
        / claims_with_citations
        if claims_with_citations
        else 0
    )

    similarities = [
        result[
            "best_passage_similarity"
        ]
        for result in claim_results
    ]

    mean_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0
    )

    cited_similarities = [
        result[
            "best_passage_similarity"
        ]
        for result in claim_results
        if result["valid_citations"]
    ]

    mean_cited_similarity = (
        sum(cited_similarities)
        / len(cited_similarities)
        if cited_similarities
        else 0
    )

    # ---------------------------------------------------------
    # 7. Save results
    # ---------------------------------------------------------

    results = {

        "document_id": DOCUMENT_ID,

        "language": "Tamil",

        "evaluation_type":
            "multilingual_passage_level_semantic_grounding",

        "embedding_model":
            MODEL_NAME,

        "grounding_threshold":
            GROUNDING_THRESHOLD,

        "passage_words":
            PASSAGE_WORDS,

        "passage_overlap":
            PASSAGE_OVERLAP,

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

        "total_claims":
            total_claims,

        "claims_with_citations":
            claims_with_citations,

        "grounded_claims":
            grounded_claims,

        "citation_coverage":
            round(
                citation_coverage,
                4
            ),

        "semantic_grounding_rate":
            round(
                semantic_grounding_rate,
                4
            ),

        "cited_claim_grounding_rate":
            round(
                cited_claim_grounding_rate,
                4
            ),

        "mean_semantic_similarity":
            round(
                mean_similarity,
                4
            ),

        "mean_cited_claim_similarity":
            round(
                mean_cited_similarity,
                4
            ),

        "evidence_matching_method":
            "all_cited_pages_overlapping_passages",

        "claim_results":
            claim_results
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
    # 8. Console output
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nTotal judgment pages     : "
        f"{total_pages}"
    )

    print(
        f"Unique citations         : "
        f"{len(unique_citations)}"
    )

    print(
        f"Valid citations          : "
        f"{len(valid_citations)}"
    )

    print(
        f"Invalid citations        : "
        f"{len(invalid_citations)}"
    )

    print(
        f"\nCitation Validity        : "
        f"{citation_validity * 100:.2f}%"
    )

    print(
        f"\nTotal claims             : "
        f"{total_claims}"
    )

    print(
        f"Claims with citations    : "
        f"{claims_with_citations}"
    )

    print(
        f"Grounded claims         : "
        f"{grounded_claims}"
    )

    print(
        f"\nCitation Coverage        : "
        f"{citation_coverage * 100:.2f}%"
    )

    print(
        f"Semantic Grounding Rate  : "
        f"{semantic_grounding_rate * 100:.2f}%"
    )

    print(
        f"Cited Claim Grounding    : "
        f"{cited_claim_grounding_rate * 100:.2f}%"
    )

    print(
        f"\nMean Passage Similarity  : "
        f"{mean_similarity:.4f}"
    )

    print(
        f"Mean Cited Similarity    : "
        f"{mean_cited_similarity:.4f}"
    )

    print(
        f"\nGrounding Threshold      : "
        f"{GROUNDING_THRESHOLD:.2f}"
    )

    # ---------------------------------------------------------
    # 9. Claim-level details
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "CLAIM-LEVEL PASSAGE GROUNDING"
    )
    print("-" * 70)

    for result in claim_results:

        print(
            f"\nClaim {result['claim_number']}:"
        )

        print(
            f"  Best evidence page    : "
            f"{result['best_evidence_page']}"
        )

        print(
            f"  Passage similarity    : "
            f"{result['best_passage_similarity']:.4f}"
        )

        print(
            f"  Grounded              : "
            f"{result['grounded']}"
        )

    # ---------------------------------------------------------
    # 10. Invalid citations
    # ---------------------------------------------------------

    if invalid_citations:

        print(
            "\nWARNING: Invalid page citations found:"
        )

        for page in invalid_citations:

            print(
                f"  [Page {page}]"
            )

    print("\nResults saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()
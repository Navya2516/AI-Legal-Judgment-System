import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "case_summaries"
    / "0adde2b14334482d83ae091f61003bc6_summary.json"
)

STRUCTURED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uploads"
    / "0adde2b14334482d83ae091f61003bc6"
    / "structured.json"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "summary_faithfulness_results.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    """
    Normalize text for simple evidence matching.
    """
    text = text.lower()

    # Remove page citation markers
    text = re.sub(r"\[page\s*\d+\]", "", text)

    # Keep alphanumeric characters and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_citations(text):
    """
    Extract citations such as [Page 12], [Page 123].
    """
    matches = re.findall(
        r"\[page\s+(\d+)\]",
        text,
        flags=re.IGNORECASE
    )

    return [int(page) for page in matches]


def split_into_claims(text):
    """
    Split summary text into sentence-level claims.
    """
    text = text.replace("\n", " ")

    sentences = re.split(r"(?<=[.!?])\s+", text)

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) < 20:
            continue

        claims.append(sentence)

    return claims


def get_page_text(structured_data, page_number):
    """
    Retrieve the original judgment text for a specific page.
    """
    for page in structured_data.get("pages", []):
        if page.get("page_number") == page_number:
            return page.get("text", "")

    return ""


def calculate_overlap(claim, evidence):
    """
    Calculate simple word overlap between a claim and page evidence.
    """

    claim_words = set(normalize_text(claim).split())
    evidence_words = set(normalize_text(evidence).split())

    if not claim_words:
        return 0.0

    common_words = claim_words.intersection(evidence_words)

    return len(common_words) / len(claim_words)


def main():

    print("=" * 65)
    print("SUMMARY FAITHFULNESS / CITATION-GROUNDING EVALUATION")
    print("=" * 65)

    summary_data = load_json(SUMMARY_FILE)
    structured_data = load_json(STRUCTURED_FILE)

    summary = summary_data["summary"]

    # Evaluate the main generated explanation
    generated_text = summary.get("simplified_explanation", "")

    if not generated_text:
        raise ValueError(
            "simplified_explanation was not found in generated summary."
        )

    print("\nGenerated summary loaded.")
    print("Original judgment loaded.")

    claims = split_into_claims(generated_text)

    print(f"\nTotal claims evaluated: {len(claims)}")

    total_claims = len(claims)
    claims_with_citations = 0
    grounded_claims = 0
    unsupported_claims = 0

    claim_results = []

    for claim in claims:

        citations = extract_citations(claim)

        if not citations:
            claims_with_citations_flag = False

            claim_results.append({
                "claim": claim,
                "citations": [],
                "citation_found": False,
                "grounded": False,
                "overlap_score": 0.0
            })

            unsupported_claims += 1
            continue

        claims_with_citations += 1

        best_overlap = 0.0
        best_page = None

        for page_number in citations:

            page_text = get_page_text(
                structured_data,
                page_number
            )

            if not page_text:
                continue

            overlap = calculate_overlap(
                claim,
                page_text
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page_number

        # Simple baseline threshold
        grounded = best_overlap >= 0.20

        if grounded:
            grounded_claims += 1
        else:
            unsupported_claims += 1

        claim_results.append({
            "claim": claim,
            "citations": citations,
            "citation_found": True,
            "grounded": grounded,
            "best_page": best_page,
            "overlap_score": round(best_overlap, 4)
        })

    citation_coverage = (
        claims_with_citations / total_claims
        if total_claims
        else 0
    )

    grounding_rate = (
        grounded_claims / total_claims
        if total_claims
        else 0
    )

    grounded_cited_claim_rate = (
        grounded_claims / claims_with_citations
        if claims_with_citations
        else 0
    )

    results = {
        "document_id": "0adde2b14334482d83ae091f61003bc6",
        "evaluation_type": "summary_faithfulness",
        "total_claims": total_claims,
        "claims_with_citations": claims_with_citations,
        "grounded_claims": grounded_claims,
        "unsupported_claims": unsupported_claims,
        "citation_coverage": round(citation_coverage, 4),
        "grounding_rate": round(grounding_rate, 4),
        "grounded_cited_claim_rate": round(
            grounded_cited_claim_rate,
            4
        ),
        "claim_results": claim_results
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

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(f"\nTotal claims             : {total_claims}")
    print(f"Claims with citations    : {claims_with_citations}")
    print(f"Grounded claims         : {grounded_claims}")
    print(f"Unsupported claims       : {unsupported_claims}")

    print(
        f"\nCitation Coverage       : "
        f"{citation_coverage:.4f} "
        f"({citation_coverage * 100:.2f}%)"
    )

    print(
        f"Grounding Rate          : "
        f"{grounding_rate:.4f} "
        f"({grounding_rate * 100:.2f}%)"
    )

    print(
        f"Grounded Cited Claims   : "
        f"{grounded_cited_claim_rate:.4f} "
        f"({grounded_cited_claim_rate * 100:.2f}%)"
    )

    print("\nResults saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()
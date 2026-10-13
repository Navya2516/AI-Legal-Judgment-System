from backend.app.services.retrieval_service import RetrievalService


INDEX_PATH = (
    "data/processed/embeddings/"
    "ravindra_kumar_2024_insc_131.index"
)

METADATA_PATH = (
    "data/processed/embeddings/"
    "ravindra_kumar_2024_insc_131_metadata.json"
)


retriever = RetrievalService(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

question = "What legal principles did the court apply regarding suppression of criminal cases?"


results = retriever.retrieve(
    question,
    top_k=5
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
    print(f"Rank       : {rank}")
    print(f"Score      : {result['score']:.4f}")
    print(f"Chunk      : {result['chunk_id']}")
    print(f"Page       : {result['page_number']}")

    print("-" * 70)

    print(result["text"])
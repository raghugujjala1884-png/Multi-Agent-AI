import chromadb

from rank_bm25 import BM25Okapi
# from langchain_ollama import OllamaEmbeddings
# from config.model_config import get_embedding_model
from config.model_config import (
    get_embedding_model,
    EMBEDDING_PROVIDER
)

from RAG.ingestion import load_pdfs, split_documents


# VECTOR_STORE_DIR = "RAG/vector_store"
# COLLECTION_NAME = "climate_knowledge"

VECTOR_STORE_DIR = "RAG/vector_store"

COLLECTION_NAMES = {
    "ollama": "climate_knowledge",
    "gemini": "climate_knowledge_gemini"
}

COLLECTION_NAME = COLLECTION_NAMES.get(EMBEDDING_PROVIDER)

if COLLECTION_NAME is None:
    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}"
    )


TOP_K = 5


# # ============================================================
# # 1. Load documents for BM25
# # ============================================================

# print("Loading documents for BM25...")

# documents = load_pdfs()
# chunks = split_documents(documents)

# print(f"Total chunks loaded: {len(chunks)}")


# # ============================================================
# # 2. Prepare BM25
# # ============================================================

# print("\nBuilding BM25 index...")

# corpus = [
#     chunk.page_content
#     for chunk in chunks
# ]

# tokenized_corpus = [
#     document.lower().split()
#     for document in corpus
# ]

# bm25 = BM25Okapi(tokenized_corpus)

# print("BM25 index ready.")


# # ============================================================
# # 3. Load embedding model
# # ============================================================

# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text"
# )


# # ============================================================
# # 4. Connect to Chroma
# # ============================================================

# client = chromadb.PersistentClient(
#     path=VECTOR_STORE_DIR
# )

# collection = client.get_collection(
#     name=COLLECTION_NAME
# )

# ============================================================
# 1. RETRIEVER STATE
# ============================================================

# These variables will be initialized only when retrieval is needed.

chunks = None
bm25 = None
embeddings = None
collection = None


# ============================================================
# 2. LAZY INITIALIZATION
# ============================================================

def _initialize_retriever():

    global chunks, bm25, embeddings, collection

    # Avoid initializing resources more than once.
    if bm25 is not None:
        return

    print("Initializing RAG retriever...")

    # Load and split PDF documents.
    documents = load_pdfs()
    loaded_chunks = split_documents(documents)

    if not loaded_chunks:
        raise FileNotFoundError(
            "No document chunks found. "
            "Check the PDF files in rag/docs."
        )

    # Build BM25 index.
    corpus = [
        chunk.page_content
        for chunk in loaded_chunks
    ]

    tokenized_corpus = [
        document.lower().split()
        for document in corpus
    ]

    loaded_bm25 = BM25Okapi(tokenized_corpus)

    # Initialize embedding model.
    loaded_embeddings = get_embedding_model()

    # Connect to ChromaDB.
    # client = chromadb.PersistentClient(
    #     path=VECTOR_STORE_DIR
    # )

    # loaded_collection = client.get_collection(
    #     name=COLLECTION_NAME
    # )

    # Connect to ChromaDB.
    client = chromadb.PersistentClient(
        path=VECTOR_STORE_DIR
    )

    loaded_collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(f"Using ChromaDB collection: {COLLECTION_NAME}")

    # Assign resources after successful initialization.
    chunks = loaded_chunks
    bm25 = loaded_bm25
    embeddings = loaded_embeddings
    collection = loaded_collection

    print(f"Total chunks loaded: {len(chunks)}")
    print("BM25 index ready.")
    print("RAG retriever initialized successfully.")



# ============================================================
# 5. Dense retrieval
# ============================================================

def dense_search(question, top_k=TOP_K):

    _initialize_retriever()

    query_embedding = embeddings.embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    dense_results = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        dense_results.append(
            {
                "chunk_id": chunk_id,
                "rank": rank,
                "document": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return dense_results


# ============================================================
# 6. Sparse retrieval - BM25
# ============================================================

def sparse_search(question, top_k=TOP_K):

    _initialize_retriever()

    tokenized_query = question.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    sparse_results = []

    for rank, index in enumerate(
        ranked_indexes,
        start=1
    ):

        sparse_results.append(
            {
                "chunk_id": f"chunk_{index}",
                "rank": rank,
                "document": chunks[index].page_content,
                "metadata": chunks[index].metadata,
                "bm25_score": float(scores[index])
            }
        )

    return sparse_results


# ============================================================
# 7. Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=60
):

    hybrid_results = {}

    # ----------------------------------------
    # Dense results
    # ----------------------------------------

    for result in dense_results:

        chunk_id = result["chunk_id"]

        hybrid_results[chunk_id] = {
            "chunk_id": chunk_id,
            "document": result["document"],
            "metadata": result["metadata"],
            "dense_rank": result["rank"],
            "dense_distance": result["distance"],
            "sparse_rank": None,
            "bm25_score": None,
            "rrf_score": 0.0
        }

        hybrid_results[chunk_id]["rrf_score"] += (
            1 / (k + result["rank"])
        )

    # ----------------------------------------
    # Sparse results
    # ----------------------------------------

    for result in sparse_results:

        chunk_id = result["chunk_id"]

        if chunk_id not in hybrid_results:

            hybrid_results[chunk_id] = {
                "chunk_id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "dense_rank": None,
                "dense_distance": None,
                "sparse_rank": result["rank"],
                "bm25_score": result["bm25_score"],
                "rrf_score": 0.0
            }

        else:

            hybrid_results[chunk_id]["sparse_rank"] = (
                result["rank"]
            )

            hybrid_results[chunk_id]["bm25_score"] = (
                result["bm25_score"]
            )

        hybrid_results[chunk_id]["rrf_score"] += (
            1 / (k + result["rank"])
        )

    # ----------------------------------------
    # Sort by RRF score
    # ----------------------------------------

    return sorted(
        hybrid_results.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )


# ============================================================
# 8. Test Hybrid Retrieval
# ============================================================

if __name__ == "__main__":

    question = (
        "What evidence shows that the global "
        "climate is warming?"
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # ----------------------------------------
    # Dense
    # ----------------------------------------

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL - CHROMA")
    print("=" * 70)

    dense_results = dense_search(
        question,
        TOP_K
    )

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        print(f"\nRank: {rank}")
        print(
            f"Source: {result['metadata']['source']}"
        )
        print(
            f"Page: {result['metadata']['page']}"
        )
        print(
            f"Dense Distance: {result['distance']}"
        )
        print(
            f"Content: {result['document'][:300]}"
        )

    # ----------------------------------------
    # Sparse
    # ----------------------------------------

    print("\n" + "=" * 70)
    print("SPARSE RETRIEVAL - BM25")
    print("=" * 70)

    sparse_results = sparse_search(
        question,
        TOP_K
    )

    for rank, result in enumerate(
        sparse_results,
        start=1
    ):

        print(f"\nRank: {rank}")
        print(
            f"Source: {result['metadata']['source']}"
        )
        print(
            f"Page: {result['metadata']['page']}"
        )
        print(
            f"BM25 Score: {result['bm25_score']}"
        )
        print(
            f"Content: {result['document'][:300]}"
        )

    # ----------------------------------------
    # Hybrid
    # ----------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL - RRF")
    print("=" * 70)

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )

    for rank, result in enumerate(
        hybrid_results[:TOP_K],
        start=1
    ):

        print(f"\nRank: {rank}")

        print(f"Chunk ID: {result['chunk_id']}")

        print(
            f"Source: {result['metadata']['source']}"
        )

        print(
            f"Page: {result['metadata']['page']}"
        )

        print(
            f"Dense Rank: {result['dense_rank']}"
        )

        print(
            f"Dense Distance: {result['dense_distance']}"
        )

        print(
            f"Sparse Rank: {result['sparse_rank']}"
        )

        print(
            f"BM25 Score: {result['bm25_score']}"
        )

        print(
            f"RRF Score: {result['rrf_score']}"
        )

        print(
            f"Content: {result['document'][:500]}"
        )
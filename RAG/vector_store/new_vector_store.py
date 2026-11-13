# from pathlib import Path

# import chromadb
# # from langchain_ollama import OllamaEmbeddings
# from config.model_config import get_embedding_model

# from RAG.ingestion import load_pdfs, split_documents


# VECTOR_STORE_DIR = Path("rag/vector_store")

# COLLECTION_NAME = "climate_knowledge"


# def main():

#     # -----------------------------------------
#     # 1. Load and split documents
#     # -----------------------------------------

#     print("Loading PDFs...")

#     documents = load_pdfs()

#     print(f"Pages loaded: {len(documents)}")

#     chunks = split_documents(documents)

#     print(f"Chunks created: {len(chunks)}")

#     # -----------------------------------------
#     # 2. Create embedding model
#     # -----------------------------------------

#     print("\nLoading embedding model...")

#     embeddings = get_embedding_model()

#     # -----------------------------------------
#     # 3. Create Chroma database
#     # -----------------------------------------

#     print("\nCreating Chroma vector store...")

#     chroma_client = chromadb.PersistentClient(
#         path=str(VECTOR_STORE_DIR)
#     )

#     collection = chroma_client.get_or_create_collection(
#         name=COLLECTION_NAME
#     )

#     # -----------------------------------------
#     # 4. Create embeddings
#     # -----------------------------------------

#     print("\nCreating embeddings...")

#     ids = [
#         f"chunk_{i}"
#         for i in range(len(chunks))
#     ]

#     texts = [
#         chunk.page_content
#         for chunk in chunks
#     ]

#     metadatas = [
#         chunk.metadata
#         for chunk in chunks
#     ]

#     vectors = embeddings.embed_documents(texts)

#     print(f"Embeddings created: {len(vectors)}")

#     # -----------------------------------------
#     # 5. Store in Chroma
#     # -----------------------------------------

#     collection.upsert(
#         ids=ids,
#         documents=texts,
#         embeddings=vectors,
#         metadatas=metadatas
#     )

#     print("\nVector store created successfully.")

#     print(f"Collection: {COLLECTION_NAME}")

#     print(f"Documents stored: {collection.count()}")


# if __name__ == "__main__":
#     main()



from pathlib import Path

import chromadb

from config.model_config import (
    get_embedding_model,
    EMBEDDING_PROVIDER
)

from RAG.ingestion import load_pdfs, split_documents


VECTOR_STORE_DIR = Path("RAG/vector_store")

COLLECTION_NAMES = {
    "ollama": "climate_knowledge",
    "gemini": "climate_knowledge_gemini"
}

COLLECTION_NAME = COLLECTION_NAMES.get(EMBEDDING_PROVIDER)

if COLLECTION_NAME is None:
    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}"
    )


def main():

    # 1. Load and split documents
    print("Loading PDFs...")

    documents = load_pdfs()
    print(f"Pages loaded: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No document chunks found.")

    # 2. Create embedding model
    print("\nLoading embedding model...")

    embeddings = get_embedding_model()

    # 3. Connect to ChromaDB
    print("\nConnecting to ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(f"Selected collection: {COLLECTION_NAME}")

    # 4. Prepare documents
    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    # 5. Generate embeddings
    print("\nCreating embeddings...")

    # vectors = embeddings.embed_documents(texts)
    import time

    vectors = []

    batch_size = 5

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        batch_vectors = embeddings.embed_documents(batch)

        vectors.extend(batch_vectors)

        print(f"Processed {len(vectors)}/{len(texts)} chunks")

        if i + batch_size < len(texts):
            time.sleep(65)

        print(f"Embeddings created: {len(vectors)}")

    # 6. Store in ChromaDB
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas
    )

    print("\nVector store created successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents stored: {collection.count()}")


if __name__ == "__main__":
    main()

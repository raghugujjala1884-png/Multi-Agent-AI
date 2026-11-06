from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings

from RAG.ingestion import load_pdfs, split_documents


VECTOR_STORE_DIR = Path("rag/vector_store")

COLLECTION_NAME = "climate_knowledge"


def main():

    # -----------------------------------------
    # 1. Load and split documents
    # -----------------------------------------

    print("Loading PDFs...")

    documents = load_pdfs()

    print(f"Pages loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # -----------------------------------------
    # 2. Create embedding model
    # -----------------------------------------

    print("\nLoading embedding model...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # -----------------------------------------
    # 3. Create Chroma database
    # -----------------------------------------

    print("\nCreating Chroma vector store...")

    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # -----------------------------------------
    # 4. Create embeddings
    # -----------------------------------------

    print("\nCreating embeddings...")

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

    vectors = embeddings.embed_documents(texts)

    print(f"Embeddings created: {len(vectors)}")

    # -----------------------------------------
    # 5. Store in Chroma
    # -----------------------------------------

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
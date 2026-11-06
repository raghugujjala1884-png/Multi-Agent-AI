import chromadb


VECTOR_STORE_DIR = "RAG/vector_store"
COLLECTION_NAME = "climate_knowledge"


client = chromadb.PersistentClient(
    path=VECTOR_STORE_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Total documents:", collection.count())

result = collection.get(
    limit=1,
    include=["documents", "metadatas", "embeddings"]
)

print("\nDocument:")
print(result["documents"][0][:500])

print("\nMetadata:")
print(result["metadatas"][0])

embedding = result["embeddings"][0]

print("\nEmbedding dimension:")
print(len(embedding))

print("\nFirst 10 embedding values:")
print(embedding[:10])
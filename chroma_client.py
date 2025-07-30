import chromadb
from chromadb.config import Settings

# New way to initialize the client
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# Create or load your collection
collection = chroma_client.get_or_create_collection(name="gemini-qa")

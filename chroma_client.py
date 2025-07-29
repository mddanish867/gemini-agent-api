import chromadb
from chromadb.config import Settings

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_store"  # You can change this path if needed
))
collection = chroma_client.get_or_create_collection(name="gemini_qa")
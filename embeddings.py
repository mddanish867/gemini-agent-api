from sentence_transformers import SentenceTransformer

# 1024-dimension model compatible with your Pinecone index
embedder = SentenceTransformer("intfloat/e5-large")

def get_embedding(text: str) -> list[float]:
    # This model expects inputs like "query: What is AI?"
    return embedder.encode(f"query: {text}").tolist()

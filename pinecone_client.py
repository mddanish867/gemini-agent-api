import os
from pinecone import Pinecone, ServerlessSpec

# Create an instance of Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Example usage: check if index exists, or create one
index_name = "gemini-qa"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Change to your actual embedding dimension
        metric='cosine',  # Or 'euclidean', 'dotproduct'
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Use your actual Pinecone project region
        )
    )

# You can now access your index
index = pc.Index(index_name)


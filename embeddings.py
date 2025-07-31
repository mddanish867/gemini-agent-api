# # embedding.py
# from sentence_transformers import SentenceTransformer
# import os

# # Hugging Face model will be cached locally in this directory (optional but recommended)
# os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

# # Load model only once
# embedder = SentenceTransformer("intfloat/e5-large")

# def get_embedding(text: str) -> list[float]:
#     return embedder.encode(f"query: {text}").tolist()

# print(f"HF_HOME is set to: {os.environ['HF_HOME']}")
# print(f"Model loaded from: {embedder.cache_folder}")



#___________________New__________________
# from sentence_transformers import SentenceTransformer
# import os
# from pathlib import Path

# # Set custom cache dir
# custom_cache_dir = os.path.join(os.getcwd(), "hf_cache")
# os.makedirs(custom_cache_dir, exist_ok=True)

# # Set env variables
# os.environ["HF_HOME"] = custom_cache_dir
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(custom_cache_dir, "transformers")

# print(f"HF_HOME is set to: {os.environ['HF_HOME']}")
# print(f"TRANSFORMERS_CACHE is set to: {os.environ['TRANSFORMERS_CACHE']}")

# # Load model
# embedder = SentenceTransformer("intfloat/e5-large")

# # Force download by running an encode
# # sample = "Hello, world!"
# # embedding = embedder.encode(f"query: {sample}")
# # print("Sample embedding generated.")


# def embedding(text: str):
#     return embedder.encode(f"query: {text}").tolist()

# # Check model path
# model_path = Path(os.environ["TRANSFORMERS_CACHE"]) / "models--intfloat--e5-large"
# print(f"Model should be cached at: {model_path}")
# print(f"Exists: {model_path.exists()}")


#--___________________New___________________
# embeddings.py
from sentence_transformers import SentenceTransformer
import os

# Set up cache directory to prevent redownload
custom_cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.makedirs(custom_cache_dir, exist_ok=True)

os.environ["HF_HOME"] = custom_cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(custom_cache_dir, "transformers")

# Load model only once
_embedder = SentenceTransformer("intfloat/e5-large")

def embedding(text: str):
    return _embedder.encode(f"query: {text}").tolist()

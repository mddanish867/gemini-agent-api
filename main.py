import uuid
import logging
from fastapi import FastAPI, HTTPException, Query
from models import QuestionRequest
from gemini_client import model
# from embeddings import get_embedding
from embeddings import embedding
from pinecone_client import index as pinecone_index
from chroma_client import collection as chroma_collection

# history implementation
from fastapi import Query
from datetime import datetime


timestamp = datetime.utcnow().isoformat()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Gemini with Chroma & Pinecone running!"}

# Debugging endpoints for and Chroma
@app.get("/chroma/debug")
def debug_chroma_data():
    results = chroma_collection.get(include=["metadatas", "ids"])
    return {
        "count": len(results["ids"]),
        "items": [
            {
                "id": results["ids"][i],
                "metadata": results["metadatas"][i]
            } for i in range(len(results["ids"]))
        ]
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question_embedding = embedding(request.question)
    try:
        # Step 1: Generate response
        response = model.generate_content(request.question)
        answer_text = response.text

        # Step 2: Generate embeddings
        question_vector = embedding(request.question)
        answer_vector = embedding(answer_text)

        question_id = str(uuid.uuid4())
        answer_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Step 3a: Store in Pinecone
        pinecone_index.upsert([
            {
                "id": question_id,
                "values": question_vector,
                "metadata": {
                    "type": "question",
                    "text": request.question,
                    "user_id": request.user_id,
                    "timestamp": timestamp
                },
            },
            {
                "id": answer_id,
                "values": answer_vector,
                "metadata": {
                    "type": "answer",
                    "text": answer_text,
                    "user_id": request.user_id,
                    "timestamp": timestamp
                },
            },
        ])

        # Step 3b: Store in Chroma
        chroma_collection.add(
    ids=[question_id, answer_id],
    embeddings=[question_vector, answer_vector],
    metadatas=[
        {
            "type": "question",
            "text": request.question,
            "user_id": request.user_id,
            "timestamp": timestamp,
        },
        {
            "type": "answer",
            "text": answer_text,
            "user_id": request.user_id,
            "timestamp": timestamp,
        },
    ]
)
        
        
        # Step 4: Return response
        logger.info(f"Question ID: {question_id}, Answer ID: {answer_id}")
        return {
            "status": "success",
            "question_id": question_id,
            "answer_id": answer_id,
            "answer": answer_text
        }
    

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search-pinecone")
async def search_pinecone(query: str = Query(...)):
    try:
        query_vector = embedding(query)
        result = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)
        return result
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/search-chroma")
async def search_chroma(query: str = Query(...)):
    try:
        query_vector = embedding(query)
        results = chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=5,
            include=["metadatas", "distances"]
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


#history logic

@app.get("/history")
async def get_user_history(user_id: str = Query(...)):
    try:
        # Fetch all questions from Chroma for the user
        results = chroma_collection.get(include=["metadatas", "ids"])
        user_questions = []

        for i, metadata in enumerate(results["metadatas"]):
            if metadata and metadata.get("user_id") == user_id and metadata.get("type") == "question":
                user_questions.append({
                    "id": results["ids"][i],
                    "question": metadata.get("text"),
                    "timestamp": metadata.get("timestamp")
                })

        # Optional: Sort by timestamp
        user_questions.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "status": "success",
            "user_id": user_id,
            "questions": user_questions
        }
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user history")

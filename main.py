# main.py (Updated)

import uuid
import logging
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from models import QuestionRequest
from gemini_client import model
from embeddings import embedding
from pinecone_client import index as pinecone_index
from chroma_client import collection as chroma_collection
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def save_qa_in_background(question: str, answer_text: str, user_id: str):
    """
    This function runs in the background. It generates embeddings and saves
    the question-answer pair to Pinecone and Chroma.
    """
    try:
        logger.info("Background task started: Generating embeddings and saving to DBs.")
        
        # Step 1: Generate embeddings for question and answer
        question_vector = embedding(question)
        answer_vector = embedding(answer_text)

        question_id = str(uuid.uuid4())
        answer_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Step 2a: Store in Pinecone (batch upsert is efficient)
        pinecone_index.upsert([
            {
                "id": question_id,
                "values": question_vector,
                "metadata": {"type": "question", "text": question, "user_id": user_id, "timestamp": timestamp},
            },
            {
                "id": answer_id,
                "values": answer_vector,
                "metadata": {"type": "answer", "text": answer_text, "user_id": user_id, "timestamp": timestamp},
            },
        ])
        logger.info(f"Saved to Pinecone with Question ID: {question_id}")

        # Step 2b: Store in Chroma
        chroma_collection.add(
            ids=[question_id, answer_id],
            embeddings=[question_vector, answer_vector],
            metadatas=[
                {"type": "question", "text": question, "user_id": user_id, "timestamp": timestamp},
                {"type": "answer", "text": answer_text, "user_id": user_id, "timestamp": timestamp},
            ]
        )
        logger.info(f"Saved to Chroma with Question ID: {question_id}")
        logger.info("Background task finished successfully.")

    except Exception as e:
        logger.error(f"Error in background task: {e}")


@app.get("/")
def root():
    return {"message": "Gemini with Chroma & Pinecone running!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    try:
        # Step 1: Generate response from Gemini first (this is the only blocking part for the user)
        response = model.generate_content(request.question)
        answer_text = response.text

        # Step 2: Add the time-consuming tasks to the background
        # These will run *after* the response is sent to the user.
        background_tasks.add_task(
            save_qa_in_background, 
            request.question, 
            answer_text, 
            request.user_id
        )

        # Step 3: Return the response IMMEDIATELY
        logger.info("Response generated. Handing off saving to background task.")
        return {
            "status": "success",                
            "user_id": request.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "answer": answer_text
        }

    except Exception as e:
        logger.error(f"Error occurred during Gemini call: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response from Gemini.")


@app.get("/history")
async def get_user_history(user_id: str = Query(...)):
    """
    Optimized history endpoint using Pinecone's `query` with a metadata filter.
    """
    try:
        # Use query() instead of get(). Provide a dummy vector and top_k.
        results = pinecone_index.query(
            vector=[0.0] * int(os.getenv("VECTOR_DIMENSION")),
            filter={"user_id": user_id,},
            top_k=100,  # Adjust as needed to retrieve all relevant history
            include_metadata=True
        )

        # The results are inside the 'matches' key.
        # Each match has a 'metadata' attribute.
        metadatas = [match['metadata'] for match in results.get('matches', [])]

        # Sort results by timestamp in descending order
        sorted_questions = sorted(
            metadatas,
            key=lambda x: x.get("timestamp", 0),
            reverse=True
        )

        return {
            "status": "success",
            "user_id": user_id,
            "questions": sorted_questions
        }
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user history")

# --- Your other endpoints (search, debug) remain the same ---

@app.get("/chroma/debug")
def debug_chroma_data():
    results = chroma_collection.get(include=["metadatas", "ids"])
    return {
        "count": chroma_collection.count(),
        "items": results
    }

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
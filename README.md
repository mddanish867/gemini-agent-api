✅ 1. How to Set Up Chroma and Pinecone
✅ 2. What the 3 endpoints in main.py are doing

✅ PART 1: Setup Instructions


🔹 A. Chroma Setup (Local DB)
ChromaDB is a lightweight, local vector database. No external service needed.

✅ Install Chroma
pip install chromadb sentence-transformers



✅ Nothing else needed.
1- Chroma will store vectors in memory or local folder (~/.chromadb/)
2-Good for prototyping, testing, or single-machine use



🔹 B. Pinecone Setup (Cloud DB)

✅ 1. Create an account:
👉 https://www.pinecone.io/
✅ 2. Create a project & environment
✅ 3. Create an index:

Go to Pinecone Console → Indexes → Create Index:

1- Name: gemini-qa
2- Dimensions: 384 (because MiniLM-L6-v2 outputs 384-d vectors)
3- Metric: cosine
4- Pods: 1 or default
5- Environment: e.g., gcp-starter



✅ 4. Add to .env:
PINECONE_API_KEY=your_api_key
PINECONE_ENV=your_env_name



✅ 5. Install Pinecone SDK:
pip install pinecone-client[pinecone]




✅ PART 2: Explanation of the 3 API Endpoints


📌 1. GET /
Purpose: Test the API is running.

GET http://localhost:8000/

Response:
{
  "message": "Gemini with Chroma & Pinecone running!"
}

✔ Verifies server is live
✔ Doesn't do any processing



📌 2. POST /ask
Purpose:

1- Accepts a user question
2- Calls Gemini API to get an answer
3- Embeds both question + answer using sentence-transformers
4- Saves the embeddings to both:

  🔹 Chroma (local)

  🔹 Pinecone (cloud)

Example Request:

POST /ask

Content-Type: application/json

{
  "question": "What is the use of AI in medicine?",
  "user_id": "user123"
}

Steps Performed:
1- Call Gemini → get answer
2- Create 384-d vector embeddings for:

   🔹Question
   🔹Answer

3- Save both as documents in:
   🔹Pinecone index (gemini-qa)
   🔹Chroma collection (gemini_qa)



📌 3. POST /search-pinecone
Purpose:
Takes a user query, converts it to vector, and searches for similar items in Pinecone

Example:

POST /search-pinecone?query=how AI helps doctors

What Happens:
1- Query is embedded as a vector
2- Pinecone index is searched for similar vectors using cosine similarity
3- Returns metadata of top k=5 matches

Response:
{
  "matches": [
    {
      "id": "answer_id_123",
      "score": 0.92,
      "metadata": {
        "type": "answer",
        "text": "AI can assist doctors in diagnosis...",
        "user_id": "user123"
      }
    }
  ]
}



📌 4. POST /search-chroma
Purpose:
Same as above — but search Chroma collection instead of Pinecone.

Example:
POST /search-chroma?query=how AI helps doctors

Response:
{
  "ids": [["question_id_xyz"]],
  "metadatas": [[
    {
      "type": "question",
      "text": "What are applications of AI in healthcare?",
      "user_id": "user123"
    }
  ]],
  "distances": [[0.2]]
}


📌 5. GET /history
Purpose:
Fetch the history of the user.

Example:
GET /history?user_id=user123

http://127.0.0.1:8000/history?user_id=user123






6- Start deevelopment server
python -m uvicorn main:app --reload





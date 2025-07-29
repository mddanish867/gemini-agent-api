from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    user_id: str | None = None


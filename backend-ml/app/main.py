from fastapi import FastAPI
from pydantic import BaseModel

from .safety import is_dangerous
from .embeddings import find_best_answer

app = FastAPI(title="TechFox NLP Service")

class ChatRequest(BaseModel):
    text: str
    user_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    status: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.text.strip()

    # Safety check
    if is_dangerous(text):
        return {
            "answer": "Man sizga bu haqida javob bera olmayman.",
            "status": "blocked"
        }

    # NLP similarity
    answer = find_best_answer(text)

    if answer is None:
        return {
            "answer": "Man bu haqida maʼlumotga ega emasman.",
            "status": "no_data"
        }

    return {
        "answer": answer,
        "status": "ok"
    }

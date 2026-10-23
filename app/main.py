from fastapi import FastAPI
from pydantic import BaseModel, Field
from retrieval.router import route_question

app = FastAPI(title = "Hospital AI Assistant")

class QuestionRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    route: str
    answer: str
    contexts: list[str] = Field(default_factory=list)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask_hospital_question(payload: QuestionRequest) -> AskResponse:
    result = route_question(payload.question)
    return AskResponse(**result)



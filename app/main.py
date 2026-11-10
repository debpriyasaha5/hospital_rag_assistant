from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval.router import route_question
from retrieval.router import clear_chat_history
from utils.logging_config import configure_logging

app = FastAPI(title="Hospital AI Assistant")
logger = configure_logging(__name__)

class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    route: str
    answer: str
    contexts: list[str] = Field(default_factory=list)


@app.get("/")
def api_root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_hospital_question(payload: QuestionRequest) -> AskResponse:
    logger.info("Received question request")
    try:
        result = route_question(payload.question, session_id=payload.session_id)
    except Exception:
        logger.exception("Question request failed")
        raise
    logger.info("Question request completed via %s route", result.get("route", "unknown"))
    return AskResponse(**result)

@app.delete("/session/{session_id}")
def clear_hospital_session(session_id: str):
    clear_chat_history(session_id)
    logger.info("Cleared conversation history")

    return {
        "status": "cleared"
    }


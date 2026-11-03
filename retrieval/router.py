from dotenv import load_dotenv

from retrieval.keyword_retriever import keyword_search
from retrieval.vector_retriever import vector_search
from retrieval.hybrid_retriever import hybrid_search
from generation.generator import generate_answer
from utils.utils import (
    _build_retrieval_query,
    _choose_route,
    _get_history,
    _is_general_question,
    _remember,
    clear_chat_history,
)

load_dotenv()

def route_question(question, session_id=None):
    if not question or not question.strip():
        return {
            "route": "general",
            "answer": "Please ask a hospital-related question."
        }

    chat_history = _get_history(session_id)

    route = _choose_route(question)
    if route == "general":
        answer = (
            "Hi! I’m your hospital assistant. Ask me about hospital services, fees, departments, or emergency information."
        )

        _remember(session_id, question, answer)

        return {
            "route": "general",
            "answer": answer
        }

    retrieval_query = _build_retrieval_query(
        question,
        chat_history
    )

    if route == "keyword":
        documents = keyword_search(retrieval_query)
    elif route == "vector":
        documents = vector_search(retrieval_query)
    else:
        documents = hybrid_search(retrieval_query)

    answer = generate_answer(question, documents, chat_history)

    _remember(session_id, question, answer)

    return {
        "route": route,
        "answer": answer,
        "contexts": [document.page_content for document in documents]
    }

if __name__ == "__main__":
    question = input("Ask a hospital question: ")
    result = route_question(question)

    print(f"Selected retrieval: {result['route']}")
    print(f"Answer: {result['answer']}")
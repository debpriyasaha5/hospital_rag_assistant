from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from retrieval.keyword_retriever import keyword_search
from retrieval.vector_retriever import vector_search
from retrieval.hybrid_retriever import hybrid_search
from generation.generator import generate_answer

load_dotenv()

class IntentDecision(BaseModel):
    intent: Literal['general', 'hospital'] = Field(
        description="Whether the user is asking a general conversation question or a hospital-related question."
    )

class RouteDecision(BaseModel):
    route: Literal["keyword", "vector", "hybrid"] = Field(
        description="The retrieval method best suited to answer the question."
    )

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Decide whether the user is asking a general conversation question
            or a hospital-related question.

            If it is a greeting, thank-you, or casual conversation, return:
            general

            If it is about hospital services, fees, departments, doctors,
            emergency, treatment, or hospital information, return:
            hospital

            Return only one word: general or hospital.
            """,
        ),
        ("human", "{question}"),
    ]
)

intent_model= init_chat_model(
    model="gpt-4o-mini", 
    temperature=0).with_structured_output(IntentDecision)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Choose the best retrieval method for the user's hospital question.

            Use keyword for questions focused on exact names, IDs, prices,
            fees, phone extensions, quantities, or other specific values.

            Use vector for a straightforward question about one topic where
            the wording may differ from the handbook.

            Use hybrid for broad, ambiguous, or multi-part questions that
            may benefit from both semantic and exact-term matching.

            Return only the selected route.
            """,
        ),
        ("human", "{question}"),
    ]
)

router_model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0,
).with_structured_output(RouteDecision)


def route_question(question):
    if not question or not question.strip():
        return {
            "route": "general",
            "answer": "Please ask a hospital-related question."
        }

    intention = (intent_prompt | intent_model).invoke(
        {"question": question}
    )

    if intention.intent == "general":
        return {
            "route": "general",
            "answer": "Hi! I’m your hospital assistant. Ask me about hospital services, fees, departments, or emergency information."
        }

    route_decision = (router_prompt | router_model).invoke(
        {"question": question}
    )

    if route_decision.route == "keyword":
        documents = keyword_search(question)
    elif route_decision.route == "vector":
        documents = vector_search(question)
    else:
        documents = hybrid_search(question)

    answer = generate_answer(question, documents)

    return {
        "route": route_decision.route,
        "answer": answer,
    }

if __name__ == "__main__":
    question = input("Ask a hospital question: ")
    result = route_question(question)

    print(f"Selected retrieval: {result['route']}")
    print(f"Answer: {result['answer']}")
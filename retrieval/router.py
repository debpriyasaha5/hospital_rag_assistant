from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from retrieval.keyword_retriever import keyword_search
from retrieval.vector_retriever import vector_search
from retrieval.hybrid_retriever import hybrid_search

load_dotenv()

class RouteDecision(BaseModel):
    route: Literal["keyword", "vector", "hybrid"] = Field(
        description="The retrieval method best suited to answer the question."
    )

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

def answer_results(question, documents):
    if not documents:
        return "I couldn't find that information in the hospital data."

    context = "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n"
        f"{document.page_content}"
        for document in documents
    )
    answer_prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the provided hospital data.
        If the data does not contain the answer, say "I don't know".
        Keep the answer concise.

        Hospital data:
        {context}

        Question:
        {question}
        """
    )
    answer_model = init_chat_model(
        model="gpt-4o",
        temperature=0,
        max_tokens=512
    )

    answer_chain = (
        answer_prompt 
        | answer_model 
        | StrOutputParser()
    )
    
    return answer_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

def route_question(question):
    decision = (router_prompt | router_model).invoke(
        {"question": question}
    )

    if decision.route == "keyword":
        documents = keyword_search(question)
        answer = answer_results(question, documents)

    elif decision.route == "vector":
        answer = vector_search(question)

    else:
        answer = hybrid_search(question)

    return {
        "route": decision.route,
        "answer": answer,
    }

if __name__ == "__main__":
    question = input("Ask a hospital question: ")
    result = route_question(question)

    print(f"Selected retrieval: {result['route']}")
    print(f"Answer: {result['answer']}")
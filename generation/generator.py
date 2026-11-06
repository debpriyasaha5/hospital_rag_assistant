from dotenv import load_dotenv
from time import perf_counter
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.logging_config import configure_logging

load_dotenv()

logger = configure_logging(__name__)

llm = init_chat_model(model="gpt-4o-mini",
                      temperature=0)


def format_text(documents):
    if not documents:
        return ""

    parts=[]
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()
        parts.append(f"Source: {source}\n{content}")
    return "\n\n".join(parts)

def format_chat_history(chat_history):
    return "\n".join(
        f"Question: {question}\nAnswer: {answer}"
        for question, answer in (chat_history or [])
    )

def generate_answer(question, documents, chat_history):
    started_at = perf_counter()
    context = format_text(documents)

    if not context:
        logger.warning("Skipping answer generation because retrieval returned no context")
        return "I couldn't find that information in the hospital data."

    prompt = ChatPromptTemplate.from_template(
        """
        You are a hospital assistant.

        Answer using only the hospital information in the context below.
        If the answer is not present in the context, say:
        "I don't know."

        Keep the answer short, clear, and helpful.

        Recent conversation:
        {chat_history}
        
        Context:
        {context}

        Question:
        {question}
        """
    )
    chain = prompt | llm | StrOutputParser()

    logger.info("Generating answer from %d retrieved documents", len(documents))
    try:
        answer = chain.invoke({
            "chat_history": format_chat_history(chat_history),
            "context": context,
            "question": question,
        })
    except Exception:
        logger.exception("LLM answer generation failed")
        raise

    logger.info("Answer generation completed in %.3f seconds", perf_counter() - started_at)
    return answer
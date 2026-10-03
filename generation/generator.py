from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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

def generate_answer(question, documents):
    context = format_text(documents)

    if not context:
        return "I couldn't find that information in the hospital data."

    prompt = ChatPromptTemplate.from_template(
        """
        You are a hospital assistant.

        Answer using only the hospital information in the context below.
        If the answer is not present in the context, say:
        "I don't know."

        Keep the answer short, clear, and helpful.

        Context:
        {context}

        Question:
        {question}
        """
    )
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": question
    })
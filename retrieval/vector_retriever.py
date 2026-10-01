from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file

ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = ROOT / "data" / "processed" / "vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"

llm = init_chat_model(model="gpt-4o", temperature=0.0, max_tokens=512)

def format_documents(documents):
    return "\n\n".join(document.page_content for document in documents)

@lru_cache(maxsize=1)
def get_vector_retriever():
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector index not found at {VECTOR_STORE_DIR}. "
            "Run ingestion/build_index.py first."
        )


    vector_store = Chroma(persist_directory=str(VECTOR_STORE_DIR), 
                          embedding_function = OpenAIEmbeddings(model="text-embedding-3-small"))
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever

def vector_search(question):
    retriever = get_vector_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context below.

        Context:
        {context}

        Question: {question}
        Answer: Make sure to answer in a concise manner and 
        if you don't know the answer, just say "I don't know"."""
        
    )

    rag_pipeline = (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),    
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_pipeline.invoke(question)

from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from retrieval.keyword_retriever import get_keyword_retriever
from retrieval.vector_retriever import get_vector_retriever


load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = ROOT / "data" / "processed" / "vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"

def format_documents(documents):
    return "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n"
        f"{document.page_content}"
        for document in documents
    )

def hybrid_search(question):
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError("Vector index not found. Run ingestion/build_index.py first.")

    keyword_retriever = get_keyword_retriever()
    vector_retriever = get_vector_retriever()

    combined_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5],
    )

    return combined_retriever.invoke(question)

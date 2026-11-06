from pathlib import Path
from functools import lru_cache
from time import perf_counter
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

from utils.logging_config import configure_logging

load_dotenv()  # Load environment variables from .env file

ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = ROOT / "data" / "processed" / "vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"

logger = configure_logging(__name__)

llm = init_chat_model(model="gpt-4o", temperature=0.0, max_tokens=512)

def format_documents(documents):
    return "\n\n".join(document.page_content for document in documents)

@lru_cache(maxsize=1)
def get_vector_retriever():
    if not VECTOR_STORE_DIR.exists():
        logger.error("Vector index is missing at %s", VECTOR_STORE_DIR)
        raise FileNotFoundError(
            f"Vector index not found at {VECTOR_STORE_DIR}. "
            "Run ingestion/build_index.py first."
        )

    logger.info("Opening vector index")
    try:
        vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        )
    except Exception:
        logger.exception("Failed to open vector index")
        raise

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info("Vector retriever is ready")
    return retriever

def vector_search(question):
    started_at = perf_counter()
    try:
        documents = get_vector_retriever().invoke(question)
    except Exception:
        logger.exception("Vector retrieval failed")
        raise

    logger.info(
        "Vector retrieval returned %d documents in %.3f seconds",
        len(documents),
        perf_counter() - started_at,
    )
    return documents

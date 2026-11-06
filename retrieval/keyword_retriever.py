from functools import lru_cache
import re
from time import perf_counter
from nltk.corpus import stopwords

from langchain_community.retrievers import BM25Retriever
from utils.logging_config import configure_logging

from ingestion.build_index import (
    load_structured_data,
    load_unstructured_data
)

logger = configure_logging(__name__)

ALIASES = {
    "lft": ("liver function test",),
    "liver function test": ("lft",),
    "kft": ("kidney function test", "rft"),
    "kidney function test": ("kft", "rft"),
    "rft": ("renal function test", "kft"),
    "renal function test": ("rft", "kft"),
    "icu": ("intensive care unit",),
    "intensive care unit": ("icu",),
    "opd": ("outpatient department",),
    "outpatient department": ("opd",),
    "ecg": ("electrocardiogram",),
    "electrocardiogram": ("ecg",),
    "price": ("cost", "fee", "fees", "charges"),
    "cost": ("price", "fee", "fees", "charges"),
    "extension": ("ext",),
    "ext": ("extension",),
}


def _tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    token_text = " ".join(tokens)
    expanded_tokens = [token for token in tokens if token not in stopwords.words('english')]

    for alias, alternatives in ALIASES.items():
        alias_tokens = alias.split()
        if re.search(rf"(?<!\w){re.escape(' '.join(alias_tokens))}(?!\w)", token_text):
            for alternative in alternatives:
                expanded_tokens.extend(
                    token for token in alternative.split()
                    if token not in set(stopwords.words('english'))
                )

    return expanded_tokens


@lru_cache(maxsize=1)
def get_keyword_retriever():
    started_at = perf_counter()
    logger.info("Building BM25 retriever")
    try:
        documents = load_unstructured_data() + load_structured_data()
    except Exception:
        logger.exception("Failed to load documents for BM25 retriever")
        raise

    if not documents:
        logger.error("No documents available to build BM25 retriever")
        raise ValueError("No documents are available for keyword search.")

    retriever = BM25Retriever.from_documents(
        documents, k=8, preprocess_func=_tokenize
    )
    logger.info(
        "Built BM25 retriever with %d documents in %.2f seconds",
        len(documents),
        perf_counter() - started_at,
    )
    return retriever

def keyword_search(question):
    started_at = perf_counter()
    try:
        documents = get_keyword_retriever().invoke(question)
    except Exception:
        logger.exception("Keyword retrieval failed")
        raise

    logger.info(
        "Keyword retrieval returned %d documents in %.3f seconds",
        len(documents),
        perf_counter() - started_at,
    )
    return documents
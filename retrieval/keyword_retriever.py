from functools import lru_cache
from pathlib import Path
import re
from time import perf_counter

import nltk
from nltk.corpus import stopwords

from langchain_community.retrievers import BM25Retriever
from utils.logging_config import configure_logging

from ingestion.build_index import (
    load_structured_data,
    load_unstructured_data
)

logger = configure_logging(__name__)

NLTK_DATA_DIRECTORY = Path.home() / "nltk_data"

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
    english_stopwords = _get_english_stopwords()
    expanded_tokens = [token for token in tokens if token not in english_stopwords]

    for alias, alternatives in ALIASES.items():
        alias_tokens = alias.split()
        if re.search(rf"(?<!\w){re.escape(' '.join(alias_tokens))}(?!\w)", token_text):
            for alternative in alternatives:
                expanded_tokens.extend(
                    token for token in alternative.split()
                    if token not in english_stopwords
                )

    return expanded_tokens


@lru_cache(maxsize=1)
def _get_english_stopwords():
    nltk_data_path = str(NLTK_DATA_DIRECTORY)
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    try:
        return set(stopwords.words("english"))
    except LookupError:
        logger.info("NLTK stopwords corpus is missing; downloading it")
        NLTK_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        downloaded = nltk.download(
            "stopwords",
            download_dir=nltk_data_path,
            quiet=True,
        )
        if not downloaded:
            logger.error("NLTK stopwords corpus download failed")
            raise RuntimeError(
                "NLTK stopwords are missing and could not be downloaded. "
                "Check network access or provision the NLTK stopwords corpus."
            )

        try:
            return set(stopwords.words("english"))
        except LookupError as error:
            logger.exception("NLTK stopwords corpus is still unavailable after download")
            raise RuntimeError(
                "NLTK reported a successful stopwords download, but the corpus "
                "could not be loaded."
            ) from error


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
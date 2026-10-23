from functools import lru_cache
import re
from nltk.corpus import stopwords

from langchain_community.retrievers import BM25Retriever

from ingestion.build_index import (
    load_structured_data,
    load_unstructured_data
)

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
    documents = load_unstructured_data() + load_structured_data()
    if not documents:
        raise ValueError("No documents are available for keyword search.")

    return BM25Retriever.from_documents(
        documents,
        k=8,
        preprocess_func=_tokenize,
    )

def keyword_search(question):
    return get_keyword_retriever().invoke(question)
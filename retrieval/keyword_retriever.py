from functools import lru_cache

from langchain_community.retrievers import BM25Retriever

from ingestion.build_index import (
    load_structured_data,
    load_unstructured_data
)

@lru_cache(maxsize=1)
def get_keyword_retriever():

    documents = load_unstructured_data() + load_structured_data()
    if not documents:
        raise ValueError("No documents are available for keyword search.")

    return BM25Retriever.from_documents(documents, k=4)

def keyword_search(question):
    return get_keyword_retriever().invoke(question)
from pathlib import Path
import csv
import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


load_dotenv()  # Load environment variables from .env file


ROOT = Path(__file__).resolve().parent.parent
HANDBOOK_FILE = ROOT / "data" / "processed" / "chunks" / "hospital_handbook_chunks.json"
STRUCTURED_FOLDER = ROOT / "data" / "processed" / "structured"
VECTOR_STORE_DIR = ROOT / "data" / "processed" / "vector_store"


def create_knowledge_base():
    all_documents = load_unstructured_data() + load_structured_data()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    return vector_store

def load_unstructured_data():
    with open(HANDBOOK_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    documents = []

    for chunk in chunks:
        documents.append(Document(
            page_content = chunk["text"],
            metadata={
                "source": chunk["source"],
                "heading": chunk["heading"],
                "chunk_id": chunk["chunk_id"],
            },
        ))

    return documents

def load_structured_data():
    documents = []

    for csv_file in STRUCTURED_FOLDER.glob("*.csv"):
        with open(csv_file, 'r', encoding="utf-8") as file:
            rows = csv.DictReader(file)

            for row_number, row in enumerate(rows):
                text_parts = []

                for column, value in row.items():
                    if value:
                        column_name = column.replace("_"," ").title()
                        text_parts.append(f"{column_name}: {value}")

                text = f"{csv_file.stem}: " + "; ".join(text_parts)
                
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(csv_file),
                            "table": csv_file.stem,
                            "row_number": row_number,
                        },
                    )
                )
    return documents


if __name__ == "__main__":
    vector_store = create_knowledge_base()
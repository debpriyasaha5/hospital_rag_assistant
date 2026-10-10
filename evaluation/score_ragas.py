import os
from pathlib import Path
import json
from datasets import Dataset
from ragas.llms import llm_factory
from ragas import evaluate
from ragas.metrics import(
        AnswerRelevancy, 
        ContextRecall,
        ContextPrecision,
        Faithfulness
    )
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


    
ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_PATH = ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = ROOT / "evaluation" / "evaluation_results.csv"

def compute_scores():
    if not EVAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {EVAL_DATA_PATH}. Please run run_eval.py first."
        )

    with open(EVAL_DATA_PATH, 'r', encoding="utf-8") as file:
        raw_data = json.load(file)

    if not raw_data:
        raise ValueError("The evaluation dataset is empty.")

    ragas_dataset = Dataset.from_dict({
        "user_input": [d["question"] for d in raw_data],
        "response": [d["answer"] for d in raw_data],
        "reference": [d["ground_truth"] for d in raw_data],
        "contexts": [d["contexts"] for d in raw_data]
    })

    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall()
    ]
    print("Evaluating Ragas Evaluation........")

    scores = evaluate(
        dataset= ragas_dataset,
        metrics=metrics,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        raise_exceptions=True,
    )
    results = scores.to_pandas()

    columns = [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    print(results[columns].to_string(index=False))

    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    compute_scores()

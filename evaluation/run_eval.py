import os
import json
from pathlib import Path

from retrieval.router import route_question

ROOT = Path(__file__).resolve().parent.parent
EVAL_QUESTION_FILE = ROOT / "evaluation" / "eval_questions.json"
EVAL_DATASET = ROOT / "evaluation" / "eval_dataset.json"

EVAL_DATA = [
    {
        "id": "Q1",
        "category": "Exact Keyword Retrieval",
        "question": "What is the contact extension for the Cardiology department?",
        "ground_truth": "The contact extension for the Cardiology department is DEPT-3101."
    },
    {
        "id": "Q2",
        "category": "Semantic Retrieval",
        "question": "Can family members visit a patient in the intensive care unit late in the evening?",
        "ground_truth": "Yes, ICU visiting hours include an evening window from 6:00 PM to 6:30 PM. Only 1 visitor per patient is permitted at a time."
    },
    {
        "id": "Q3",
        "category": "Hybrid / Structured",
        "question": "How much does the Senior Citizen Checkup cost and what fasting is required?",
        "ground_truth": "The Senior Citizen Checkup costs ₹7500 and requires 10-12 hours of fasting."
    },
    {
        "id": "Q4",
        "category": "Insurance & Billing",
        "question": "What happens if my insurance provider is not empanelled with the hospital?",
        "ground_truth": "You must pay the bill directly upfront and submit a reimbursement claim to your insurer later using the itemized final bill, discharge summary, and reports provided by the billing department."
    },
    {
        "id": "Q5",
        "category": "General Policy",
        "question": "What is the time required for processing a discharge after the doctor signs off?",
        "ground_truth": "Discharge processing takes roughly 2 to 4 hours due to billing finalization and pharmacy reconciliation."
    },
    {
        "id": "Q6",
        "category": "Exact Keyword Retrieval",
        "question": "What is the contact extension for the Neurology department?",
        "ground_truth": "The contact extension for the Neurology department is DEPT-3201."
    },
    {
        "id": "Q7",
        "category": "General Information",
        "question": "What are the OPD registration hours for the hospital?",
        "ground_truth": "The OPD registration counter operates from 8:00 AM to 8:00 PM daily."
    },
    {
        "id": "Q8",
        "category": "Emergency Services",
        "question": "Are ambulance services available for emergency transport?",
        "ground_truth": "Yes, the hospital provides emergency ambulance services including BLS, ALS, ICU-equipped, and neonatal transport vehicles."
    },
    {
        "id": "Q9",
        "category": "Policy / Visitor Rules",
        "question": "Can a patient's family stay overnight in a general ward?",
        "ground_truth": "No, overnight stays in the general ward are not allowed under normal visiting rules; visitors are restricted to the listed visiting hours."
    },
    {
        "id": "Q10",
        "category": "Appointment Scheduling",
        "question": "How can I book an appointment with a doctor?",
        "ground_truth": "Appointments can be booked through the hospital helpdesk, patient portal/app, or at the OPD registration counter, depending on the doctor's listed availability."
    }
]

def generate_eval_dataset():
    with open(EVAL_QUESTION_FILE, 'w', encoding="utf-8") as file:
        json.dump(EVAL_DATA, file, indent=4)
    print(f"✅ Created evaluation benchmark dataset at {EVAL_QUESTION_FILE}")

def run_evaluation():
    if not EVAL_QUESTION_FILE.exists():
        raise FileNotFoundError(f"Missing {EVAL_QUESTION_FILE}. Please create the file first.")

    with open(EVAL_QUESTION_FILE, 'r', encoding="utf-8") as file:
        questions_data = json.load(file)

    eval_samples = []
    print("Running RAG evaluation over dataset...")

    for item in questions_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        rag_ouput = route_question(question)

        eval_samples.append({
            "question": question,
            "answer": rag_ouput["answer"],
            "contexts": rag_ouput["contexts"],
            "ground_truth": ground_truth,
            "category": item.get("category", "General")
        })

    with open(EVAL_DATASET, 'w', encoding="utf-8") as file:
        json.dump(eval_samples, file, indent=4)

    print(f"\n Completed! Saved {len(eval_samples)} evaluate-ready records to: {EVAL_DATASET}")



if __name__=="__main__":
    generate_eval_dataset()
    run_evaluation()
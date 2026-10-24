
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig

from generator import generate_answer
from retrieval import search_documents


load_dotenv()

DATASET_FILE = Path("evaluation_dataset.json")
OUTPUT_FILE = Path("evaluation_results.csv")

GEMINI_MODEL = "gemini-3.5-flash-lite"


def load_evaluation_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            "evaluation_dataset.json was not found."
        )

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not data:
        raise ValueError("The evaluation dataset is empty.")

    for index, item in enumerate(data, start=1):
        for field in ("question", "ground_truth"):
            if not item.get(field):
                raise ValueError(
                    f"Entry {index} is missing '{field}'."
                )

        if "Replace with" in item["ground_truth"]:
            raise ValueError(
                f"Replace the placeholder answer in entry {index}."
            )

    return data


def run_evaluation():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in your .env file."
        )

    evaluation_data = load_evaluation_dataset()

    # Configure Gemini as the RAGAS evaluation language model.
    evaluator_chat_model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
    )

    evaluator_llm = LangchainLLMWrapper(
        evaluator_chat_model
    )

    # Configure Google embeddings for answer relevancy.
    evaluator_embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        evaluator_embedding_model
    )

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(
        f"Evaluating {len(evaluation_data)} questions...\n"
    )

    # Generate answers and collect retrieved context.
    for index, item in enumerate(evaluation_data, start=1):
        question = item["question"]

        print(
            f"[{index}/{len(evaluation_data)}] {question}"
        )

        retrieved = search_documents(
            question,
            top_k=5,
        )

        context = [
            match["text"]
            for match in retrieved
        ]

        if not context:
            print("Warning: no passages retrieved.")

        answer = generate_answer(
            question,
            top_k=5,
        )

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

        print(f"Retrieved passages: {len(context)}")
        print(f"Answer: {answer[:250]}...\n")

    # Build the dataset expected by RAGAS.
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Running RAGAS metrics with Gemini...")

    # Limit concurrency and retry transient failures.
    run_config = RunConfig(
        max_workers=2,
        timeout=180,
        max_retries=3,
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    # Save evaluation results.
    results_df = results.to_pandas()
    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nEvaluation complete!")
    print(f"Results saved to: {OUTPUT_FILE.resolve()}")

    metric_columns = [
        column
        for column in ("faithfulness", "answer_relevancy")
        if column in results_df.columns
    ]

    if metric_columns:
        print("\nAverage scores:")
        print(
            results_df[metric_columns]
            .mean()
            .round(3)
        )

        print("\nValid scores per metric:")
        print(
            results_df[metric_columns]
            .count()
        )

    print("\nPer-question results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    try:
        run_evaluation()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    except Exception as error:
        print(f"\nEvaluation failed: {error}")
        print(
            "\nCheck your Gemini API key, model availability, "
            "internet connection, RAGAS version, and package "
            "compatibility."
        )

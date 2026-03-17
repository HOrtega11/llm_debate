
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from debaterA import DebaterA
from debaterB import DebaterB
from judge import Judge
from debate import run_debate
from evaluation import (
    compute_accuracy,
    save_json,
    evaluate_and_save_all
)

#Workflow in terminal: 
#cd llm_debate
#source venv/bin/activate
#python main.py




def load_questions(file_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Load questions from a local JSON file.

    Expected format:
    [
        {
            "id": 1,
            "question": "Did the Roman Empire exist at the same time as the Mayan civilization?",
            "answer": "yes"
        },
        ...
    ]
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of question objects.")

    cleaned = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        qid = item.get("id", i + 1)
        question = item.get("question")
        answer = item.get("answer")

        if question is None or answer is None:
            continue

        cleaned.append({
            "id": qid,
            "question": str(question),
            "answer": str(answer).strip().lower()
        })

    return cleaned[:limit]


def majority_vote(answers: List[str]) -> str:
    """
    Majority vote helper for self-consistency baseline.
    """
    counts = {}
    for ans in answers:
        ans = str(ans).strip().lower()
        counts[ans] = counts.get(ans, 0) + 1

    if not counts:
        return "unknown"

    return max(counts, key=counts.get)


def run_direct_qa_baseline(
    questions: List[Dict[str, Any]],
    model_name: str
) -> List[Dict[str, Any]]:
    """
    Baseline 1: Direct QA
    Same LLM answers the question directly with reasoning, no debate.
    """
    debater = DebaterA(model_name=model_name)
    results = []

    for item in questions:
        question_id = item["id"]
        question_text = item["question"]
        gold_answer = item["answer"]

        response = debater.direct_answer(question_text, temperature=0.2)

        results.append({
            "id": question_id,
            "question": question_text,
            "gold_answer": gold_answer,
            "prediction": response.get("answer", "unknown"),
            "reasoning": response.get("reasoning", ""),
            "raw_response": response.get("raw_response", "")
        })

    return results


def run_self_consistency_baseline(
    questions: List[Dict[str, Any]],
    model_name: str,
    num_samples: int
) -> List[Dict[str, Any]]:
    """
    Baseline 2: Self-Consistency
    Sample N direct answers from the same model and take majority vote.

    N should match the total number of LLM calls in the debate pipeline:
      2 initial positions + (2 * num_rounds) + 1 judge
    """
    debater = DebaterA(model_name=model_name)
    results = []

    for item in questions:
        question_id = item["id"]
        question_text = item["question"]
        gold_answer = item["answer"]

        sampled_answers = []
        sampled_reasoning = []
        sampled_raw = []

        for _ in range(num_samples):
            response = debater.direct_answer(question_text, temperature=0.7)
            sampled_answers.append(response.get("answer", "unknown"))
            sampled_reasoning.append(response.get("reasoning", ""))
            sampled_raw.append(response.get("raw_response", ""))

        final_prediction = majority_vote(sampled_answers)

        results.append({
            "id": question_id,
            "question": question_text,
            "gold_answer": gold_answer,
            "prediction": final_prediction,
            "sampled_answers": sampled_answers,
            "sampled_reasoning": sampled_reasoning,
            "sampled_raw_responses": sampled_raw
        })

    return results


def run_debate_pipeline(
    questions: List[Dict[str, Any]],
    debater_model: str,
    judge_model: str,
    num_rounds: int = 3
) -> List[Dict[str, Any]]:
    """
    Full Debate + Judge pipeline.
    """
    debater_a = DebaterA(model_name=debater_model)
    debater_b = DebaterB(model_name=debater_model)
    judge = Judge(model_name=judge_model)

    results = []

    for item in questions:
        question_id = item["id"]
        question_text = item["question"]
        gold_answer = item["answer"]

        debate_result = run_debate(
            question=question_text,
            gold_answer=gold_answer,
            debater_a=debater_a,
            debater_b=debater_b,
            judge=judge,
            num_rounds=num_rounds
        )

        results.append({
            "id": question_id,
            "question": question_text,
            "gold_answer": gold_answer,
            "initial_positions": debate_result.get("initial_positions", {}),
            "transcript": debate_result.get("transcript", []),
            "judge_result": debate_result.get("judge_result", {}),
            "prediction": debate_result.get("final_answer", "unknown"),
            "consensus_reached": debate_result.get("consensus_reached", False),
            "stopped_early": debate_result.get("stopped_early", False)
        })

    return results


def print_summary(name: str, results: List[Dict[str, Any]]) -> None:
    acc = compute_accuracy(results)
    total = len(results)
    correct = sum(
        1 for r in results
        if str(r.get("prediction", "")).strip().lower() ==
           str(r.get("gold_answer", "")).strip().lower()
    )

    print(f"{name}: {correct}/{total} correct | accuracy = {acc:.3f}")


def main() -> None:
    # -----------------------------
    # Configuration
    # -----------------------------
    dataset_path = "data/questions.json"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_questions = 100
    num_rounds = 3

    debater_model = "gpt-4.1-mini"
    judge_model = "gpt-4.1-mini"

    # Per assignment:
    # Self-consistency sample count should match total LLM calls in debate.
    # 2 initial + (2 * num_rounds) + 1 judge
    self_consistency_samples = 2 + (2 * num_rounds) + 1

    # -----------------------------
    # Optional environment check
    # -----------------------------
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set.")
        print("Set it before running if your client requires it.\n")

    # -----------------------------
    # Load dataset
    # -----------------------------
    questions = load_questions(dataset_path, limit=num_questions)
    print(f"Loaded {len(questions)} questions from {dataset_path}")

    # Save run config
    run_config = {
        "dataset_path": dataset_path,
        "num_questions": len(questions),
        "num_rounds": num_rounds,
        "debater_model": debater_model,
        "judge_model": judge_model,
        "self_consistency_samples": self_consistency_samples
    }
    save_json(output_dir / "run_config.json", run_config)

    # -----------------------------
    # Baseline 1: Direct QA
    # -----------------------------
    print("\nRunning Direct QA baseline...")
    direct_results = run_direct_qa_baseline(
        questions=questions,
        model_name=debater_model
    )
    print_summary("Direct QA", direct_results)

    # -----------------------------
    # Baseline 2: Self-Consistency
    # -----------------------------
    print("\nRunning Self-Consistency baseline...")
    sc_results = run_self_consistency_baseline(
        questions=questions,
        model_name=debater_model,
        num_samples=self_consistency_samples
    )
    print_summary("Self-Consistency", sc_results)

    # -----------------------------
    # Debate + Judge
    # -----------------------------
    print("\nRunning Debate + Judge pipeline...")
    debate_results = run_debate_pipeline(
        questions=questions,
        debater_model=debater_model,
        judge_model=judge_model,
        num_rounds=num_rounds
    )
    print_summary("Debate + Judge", debate_results)

    # -----------------------------
    # Evaluation + Saving
    # -----------------------------
    metrics = evaluate_and_save_all(
        output_dir=output_dir,
        direct_results=direct_results,
        self_consistency_results=sc_results,
        debate_results=debate_results
    )

    print("\nSaved outputs to:", output_dir.resolve())
    print("\nMetrics summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()









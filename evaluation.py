

import json
from pathlib import Path
from typing import List, Dict, Any


def normalize_answer(answer: Any) -> str:
    """
    Normalize answers so evaluation is consistent.
    Expected task labels are usually yes/no.
    """

    if answer is None:
        return "unknown"

    answer = str(answer).strip().lower()

    if answer in {"yes", "true", "1"}:
        return "yes"
    if answer in {"no", "false", "0"}:
        return "no"

    return answer


def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Compare prediction vs ground-truth answer.

    Each result item is expected to have:
      - "prediction"
      - "gold_answer"

    Returns:
      float accuracy in [0,1]
    """

    if not results:
        return 0.0

    correct = 0
    total = 0

    for item in results:
        pred = normalize_answer(item.get("prediction"))
        gold = normalize_answer(item.get("gold_answer"))

        if gold == "unknown":
            continue

        total += 1
        if pred == gold:
            correct += 1

    if total == 0:
        return 0.0

    return correct / total


def add_correctness_flags(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add an 'is_correct' field to each result row.
    Useful for later analysis and reporting.
    """

    updated = []

    for item in results:
        pred = normalize_answer(item.get("prediction"))
        gold = normalize_answer(item.get("gold_answer"))

        new_item = dict(item)
        new_item["normalized_prediction"] = pred
        new_item["normalized_gold_answer"] = gold
        new_item["is_correct"] = (pred == gold)

        updated.append(new_item)

    return updated


def save_json(path: Any, data: Any) -> None:
    """
    Save Python object as pretty JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Any, rows: List[Dict[str, Any]]) -> None:
    """
    Save list of dicts as JSONL, one JSON object per line.
    Good for logging all intermediate data.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_results(name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a compact summary for one experiment setting.
    """

    accuracy = compute_accuracy(results)
    num_examples = len(results)
    num_correct = sum(
        1 for item in results
        if normalize_answer(item.get("prediction")) == normalize_answer(item.get("gold_answer"))
    )

    return {
        "system": name,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": accuracy
    }


def compare_systems(
    direct_results: List[Dict[str, Any]],
    self_consistency_results: List[Dict[str, Any]],
    debate_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare the required systems for the assignment:

    1. Direct QA
    2. Self-Consistency
    3. Debate + Judge

    Returns:
      dict summary
    """

    return {
        "direct_qa": summarize_results("Direct QA", direct_results),
        "self_consistency": summarize_results("Self-Consistency", self_consistency_results),
        "debate_judge": summarize_results("Debate + Judge", debate_results)
    }


def prepare_debate_records_for_logging(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Record all intermediate data required by the assignment.

    Each debate result should ideally include:
      - question
      - gold_answer
      - initial_positions
      - transcript
      - judge_result
      - prediction

    This function ensures the saved records are standardized.
    """

    records = []

    for item in results:
        record = {
            "id": item.get("id"),
            "question": item.get("question"),
            "gold_answer": item.get("gold_answer"),
            "prediction": item.get("prediction"),
            "is_correct": normalize_answer(item.get("prediction")) == normalize_answer(item.get("gold_answer")),
            "initial_positions": item.get("initial_positions", {}),
            "transcript": item.get("transcript", []),
            "judge_result": item.get("judge_result", {}),
            "consensus_reached": item.get("consensus_reached", False),
            "stopped_early": item.get("stopped_early", False)
        }
        records.append(record)

    return records


def evaluate_and_save_all(
    output_dir: Any,
    direct_results: List[Dict[str, Any]],
    self_consistency_results: List[Dict[str, Any]],
    debate_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Full evaluation helper.

    Saves:
      - direct QA detailed results
      - self-consistency detailed results
      - debate detailed results
      - debate JSONL logs with intermediate data
      - metrics summary

    Returns:
      metrics summary dict
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_results_flagged = add_correctness_flags(direct_results)
    sc_results_flagged = add_correctness_flags(self_consistency_results)
    debate_results_flagged = add_correctness_flags(debate_results)

    debate_records = prepare_debate_records_for_logging(debate_results_flagged)
    metrics = compare_systems(
        direct_results=direct_results_flagged,
        self_consistency_results=sc_results_flagged,
        debate_results=debate_results_flagged
    )

    save_json(output_dir / "direct_qa_results.json", direct_results_flagged)
    save_json(output_dir / "self_consistency_results.json", sc_results_flagged)
    save_json(output_dir / "debate_results.json", debate_results_flagged)
    save_jsonl(output_dir / "debate_records.jsonl", debate_records)
    save_json(output_dir / "metrics.json", metrics)

    return metrics






















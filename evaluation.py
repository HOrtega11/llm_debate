

import csv
import json
from math import comb
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from scipy.stats import chi2


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
    Add normalized fields and an is_correct field to each result row.
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Any, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Any, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_results(name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    accuracy = compute_accuracy(results)
    valid_items = [
        item for item in results
        if normalize_answer(item.get("gold_answer")) != "unknown"
    ]
    num_examples = len(valid_items)
    num_correct = sum(
        1 for item in valid_items
        if normalize_answer(item.get("prediction")) == normalize_answer(item.get("gold_answer"))
    )

    return {
        "system": name,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": accuracy
    }


def _index_by_id(results: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """
    Build lookup by question id for paired comparisons.
    """
    return {item.get("id"): item for item in results if item.get("id") is not None}


def _paired_correctness_vector(results: List[Dict[str, Any]]) -> Dict[Any, int]:
    """
    Returns a dict: question_id -> 1 if correct else 0
    Only includes rows with known gold answers.
    """
    out = {}
    for item in results:
        qid = item.get("id")
        if qid is None:
            continue

        gold = normalize_answer(item.get("gold_answer"))
        if gold == "unknown":
            continue

        pred = normalize_answer(item.get("prediction"))
        out[qid] = 1 if pred == gold else 0

    return out


def mcnemar_exact_test(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    name_a: str,
    name_b: str
) -> Dict[str, Any]:
    """
    Exact McNemar test for paired binary correctness outcomes.

    b = A correct, B wrong
    c = A wrong, B correct

    Two-sided exact p-value is computed from Binomial(n=b+c, p=0.5).
    """
    by_id_a = _index_by_id(results_a)
    by_id_b = _index_by_id(results_b)

    common_ids = sorted(set(by_id_a.keys()) & set(by_id_b.keys()))

    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    both_correct = 0
    both_wrong = 0

    for qid in common_ids:
        a_item = by_id_a[qid]
        b_item = by_id_b[qid]

        gold_a = normalize_answer(a_item.get("gold_answer"))
        gold_b = normalize_answer(b_item.get("gold_answer"))

        if gold_a == "unknown" or gold_b == "unknown":
            continue

        a_correct = normalize_answer(a_item.get("prediction")) == gold_a
        b_correct = normalize_answer(b_item.get("prediction")) == gold_b

        if a_correct and not b_correct:
            a_correct_b_wrong += 1
        elif not a_correct and b_correct:
            a_wrong_b_correct += 1
        elif a_correct and b_correct:
            both_correct += 1
        else:
            both_wrong += 1

    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        p_value = 1.0
    else:
        k = min(a_correct_b_wrong, a_wrong_b_correct)
        tail_prob = sum(comb(discordant, i) for i in range(0, k + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail_prob)

    return {
        "system_a": name_a,
        "system_b": name_b,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "p_value_exact_mcnemar": p_value
    }


def cochran_q_test(
    direct_results: List[Dict[str, Any]],
    self_consistency_results: List[Dict[str, Any]],
    debate_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Cochran's Q test for 3 paired classifiers on the same items.
    Returns Q statistic and chi-square p-value with df = k - 1.
    """
    direct_vec = _paired_correctness_vector(direct_results)
    sc_vec = _paired_correctness_vector(self_consistency_results)
    debate_vec = _paired_correctness_vector(debate_results)

    common_ids = sorted(set(direct_vec.keys()) & set(sc_vec.keys()) & set(debate_vec.keys()))

    if not common_ids:
        return {
            "num_items": 0,
            "num_systems": 3,
            "q_statistic": None,
            "degrees_of_freedom": 2,
            "p_value": None,
            "systems": ["Direct QA", "Self-Consistency", "Debate + Judge"],
            "note": "No common question IDs available across all three systems."
        }

    matrix = []
    for qid in common_ids:
        matrix.append([
            direct_vec[qid],
            sc_vec[qid],
            debate_vec[qid]
        ])

    n = len(matrix)
    k = 3

    col_sums = [sum(row[j] for row in matrix) for j in range(k)]
    row_sums = [sum(row) for row in matrix]

    sum_col_sq = sum(c ** 2 for c in col_sums)
    sum_row_sq = sum(r ** 2 for r in row_sums)
    total_sum = sum(col_sums)

    numerator = (k - 1) * (k * sum_col_sq - total_sum ** 2)
    denominator = k * total_sum - sum_row_sq

    if denominator == 0:
        q_stat = 0.0
    else:
        q_stat = numerator / denominator

    p_value = chi2.sf(q_stat, k - 1)

    return {
        "num_items": n,
        "num_systems": k,
        "q_statistic": q_stat,
        "degrees_of_freedom": k - 1,
        "p_value": p_value,
        "systems": ["Direct QA", "Self-Consistency", "Debate + Judge"]
    }


def prepare_debate_records_for_logging(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Record all intermediate data required by the assignment.
    """
    records = []

    for item in results:
        judge_result = item.get("judge_result", {})

        record = {
            "id": item.get("id"),
            "question": item.get("question"),
            "gold_answer": item.get("gold_answer"),
            "prediction": item.get("prediction"),
            "normalized_prediction": normalize_answer(item.get("prediction")),
            "normalized_gold_answer": normalize_answer(item.get("gold_answer")),
            "is_correct": normalize_answer(item.get("prediction")) == normalize_answer(item.get("gold_answer")),
            "initial_positions": item.get("initial_positions", {}),
            "transcript": item.get("transcript", []),
            "judge_result": judge_result,
            "judge_analysis": judge_result.get("analysis", ""),
            "judge_winner": judge_result.get("winner", ""),
            "judge_verdict": judge_result.get("verdict", ""),
            "judge_confidence": judge_result.get("confidence", ""),
            "consensus_reached": item.get("consensus_reached", False),
            "stopped_early": item.get("stopped_early", False)
        }
        records.append(record)

    return records


def system_summary_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "system": metrics["direct_qa"]["system"],
            "num_examples": metrics["direct_qa"]["num_examples"],
            "num_correct": metrics["direct_qa"]["num_correct"],
            "accuracy": round(metrics["direct_qa"]["accuracy"], 4),
        },
        {
            "system": metrics["self_consistency"]["system"],
            "num_examples": metrics["self_consistency"]["num_examples"],
            "num_correct": metrics["self_consistency"]["num_correct"],
            "accuracy": round(metrics["self_consistency"]["accuracy"], 4),
        },
        {
            "system": metrics["debate_judge"]["system"],
            "num_examples": metrics["debate_judge"]["num_examples"],
            "num_correct": metrics["debate_judge"]["num_correct"],
            "accuracy": round(metrics["debate_judge"]["accuracy"], 4),
        },
    ]


def significance_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    sig = metrics.get("significance_tests", {})
    rows = []

    cochran = sig.get("cochrans_q", {})
    rows.append({
        "test": "Cochran's Q",
        "comparison": "Direct QA vs Self-Consistency vs Debate + Judge",
        "statistic": cochran.get("q_statistic"),
        "df": cochran.get("degrees_of_freedom"),
        "p_value": cochran.get("p_value"),
    })

    for key in [
        "debate_vs_direct_qa",
        "debate_vs_self_consistency",
        "self_consistency_vs_direct_qa",
    ]:
        test = sig.get(key, {})
        rows.append({
            "test": "McNemar exact",
            "comparison": f"{test.get('system_a', '')} vs {test.get('system_b', '')}",
            "statistic": test.get("discordant_pairs"),
            "df": "",
            "p_value": test.get("p_value_exact_mcnemar"),
        })

    return rows


def get_judge_confidences(debate_results: List[Dict[str, Any]]) -> List[int]:
    vals = []
    for row in debate_results:
        judge = row.get("judge_result", {})
        conf = judge.get("confidence", None)
        try:
            conf_int = int(conf)
            if 1 <= conf_int <= 5:
                vals.append(conf_int)
        except Exception:
            pass
    return vals


def build_improvement_rows(
    direct_results: List[Dict[str, Any]],
    self_consistency_results: List[Dict[str, Any]],
    debate_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cases where Debate + Judge was correct and at least one baseline was wrong.
    """
    direct_by_id = _index_by_id(direct_results)
    sc_by_id = _index_by_id(self_consistency_results)
    debate_by_id = _index_by_id(debate_results)

    common_ids = sorted(set(direct_by_id) & set(sc_by_id) & set(debate_by_id))
    rows = []

    for qid in common_ids:
        d = direct_by_id[qid]
        s = sc_by_id[qid]
        b = debate_by_id[qid]

        gold = normalize_answer(b.get("gold_answer"))
        direct_pred = normalize_answer(d.get("prediction"))
        sc_pred = normalize_answer(s.get("prediction"))
        debate_pred = normalize_answer(b.get("prediction"))

        direct_correct = direct_pred == gold
        sc_correct = sc_pred == gold
        debate_correct = debate_pred == gold

        if debate_correct and (not direct_correct or not sc_correct):
            rows.append({
                "id": qid,
                "question": b.get("question", ""),
                "gold_answer": gold,
                "direct_prediction": direct_pred,
                "self_consistency_prediction": sc_pred,
                "debate_prediction": debate_pred,
                "direct_correct": direct_correct,
                "self_consistency_correct": sc_correct,
                "debate_correct": debate_correct,
                "consensus_reached": b.get("consensus_reached", False),
                "stopped_early": b.get("stopped_early", False),
            })

    return rows


def markdown_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    if not rows:
        return "No data available.\n"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def write_blog_tables(
    output_path: Path,
    metrics: Dict[str, Any],
    direct_results: List[Dict[str, Any]],
    sc_results: List[Dict[str, Any]],
    debate_results: List[Dict[str, Any]],
) -> None:
    system_rows = system_summary_rows(metrics)
    sig_rows = significance_rows(metrics)
    improvement_rows = build_improvement_rows(direct_results, sc_results, debate_results)

    consensus_count = sum(1 for r in debate_results if r.get("consensus_reached", False))
    early_stop_count = sum(1 for r in debate_results if r.get("stopped_early", False))
    total = len(debate_results)

    text = []
    text.append("# Blog Tables\n")
    text.append("## Table 1. Main Accuracy Results\n")
    text.append(markdown_table(
        system_rows,
        ["system", "num_examples", "num_correct", "accuracy"]
    ))

    text.append("\n## Table 2. Significance Tests\n")
    text.append(markdown_table(
        sig_rows,
        ["test", "comparison", "statistic", "df", "p_value"]
    ))

    text.append("\n## Table 3. Debate Dynamics\n")
    text.append(markdown_table(
        [{
            "total_questions": total,
            "consensus_reached": consensus_count,
            "stopped_early": early_stop_count,
            "consensus_rate": round(consensus_count / total, 4) if total else 0.0,
            "early_stop_rate": round(early_stop_count / total, 4) if total else 0.0,
        }],
        ["total_questions", "consensus_reached", "stopped_early", "consensus_rate", "early_stop_rate"]
    ))

    text.append("\n## Table 4. Cases Where Debate Helped\n")
    if improvement_rows:
        preview_rows = improvement_rows[:10]
        text.append(markdown_table(
            preview_rows,
            [
                "id",
                "gold_answer",
                "direct_prediction",
                "self_consistency_prediction",
                "debate_prediction",
                "direct_correct",
                "self_consistency_correct",
                "debate_correct",
            ]
        ))
        text.append("\nOnly the first 10 improvement cases are shown here; full cases are saved in `improvement_cases.csv`.\n")
    else:
        text.append("No cases found where Debate + Judge improved over at least one baseline.\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))


def make_accuracy_bar(metrics: Dict[str, Any], fig_path: Path) -> None:
    labels = ["Direct QA", "Self-Consistency", "Debate + Judge"]
    values = [
        metrics["direct_qa"]["accuracy"],
        metrics["self_consistency"]["accuracy"],
        metrics["debate_judge"]["accuracy"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by System")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def make_judge_confidence_hist(debate_results: List[Dict[str, Any]], fig_path: Path) -> None:
    confidences = get_judge_confidences(debate_results)

    plt.figure(figsize=(7, 5))
    if confidences:
        plt.hist(confidences, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.9)
        plt.xticks([1, 2, 3, 4, 5])
    else:
        plt.text(0.5, 0.5, "No judge confidence values found", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.xlabel("Judge Confidence")
    plt.ylabel("Count")
    plt.title("Distribution of Judge Confidence")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def make_consensus_bar(debate_results: List[Dict[str, Any]], fig_path: Path) -> None:
    consensus_yes = sum(1 for r in debate_results if r.get("consensus_reached", False))
    consensus_no = len(debate_results) - consensus_yes

    plt.figure(figsize=(6, 5))
    plt.bar(["Consensus", "No Consensus"], [consensus_yes, consensus_no])
    plt.ylabel("Count")
    plt.title("Immediate Consensus in Debate Initialization")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def compare_systems(
    direct_results: List[Dict[str, Any]],
    self_consistency_results: List[Dict[str, Any]],
    debate_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare Direct QA, Self-Consistency, and Debate + Judge.
    Includes significance tests.
    """
    return {
        "direct_qa": summarize_results("Direct QA", direct_results),
        "self_consistency": summarize_results("Self-Consistency", self_consistency_results),
        "debate_judge": summarize_results("Debate + Judge", debate_results),
        "significance_tests": {
            "cochrans_q": cochran_q_test(
                direct_results,
                self_consistency_results,
                debate_results
            ),
            "debate_vs_direct_qa": mcnemar_exact_test(
                debate_results, direct_results, "Debate + Judge", "Direct QA"
            ),
            "debate_vs_self_consistency": mcnemar_exact_test(
                debate_results, self_consistency_results, "Debate + Judge", "Self-Consistency"
            ),
            "self_consistency_vs_direct_qa": mcnemar_exact_test(
                self_consistency_results, direct_results, "Self-Consistency", "Direct QA"
            )
        }
    }


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
      - metrics summary including significance tests
      - CSV tables
      - Markdown blog tables
      - PNG figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    direct_results_flagged = add_correctness_flags(direct_results)
    sc_results_flagged = add_correctness_flags(self_consistency_results)
    debate_results_flagged = add_correctness_flags(debate_results)

    debate_records = prepare_debate_records_for_logging(debate_results_flagged)

    metrics = compare_systems(
        direct_results=direct_results_flagged,
        self_consistency_results=sc_results_flagged,
        debate_results=debate_results_flagged
    )

    # Save raw/detailed outputs
    save_json(output_dir / "direct_qa_results.json", direct_results_flagged)
    save_json(output_dir / "self_consistency_results.json", sc_results_flagged)
    save_json(output_dir / "debate_results.json", debate_results_flagged)
    save_jsonl(output_dir / "debate_records.jsonl", debate_records)
    save_json(output_dir / "metrics.json", metrics)

    # Save report tables
    accuracy_rows = system_summary_rows(metrics)
    sig_rows = significance_rows(metrics)
    improvement_rows = build_improvement_rows(
        direct_results_flagged,
        sc_results_flagged,
        debate_results_flagged
    )

    save_csv(
        tables_dir / "accuracy_table.csv",
        accuracy_rows,
        ["system", "num_examples", "num_correct", "accuracy"]
    )

    save_csv(
        tables_dir / "significance_table.csv",
        sig_rows,
        ["test", "comparison", "statistic", "df", "p_value"]
    )

    save_csv(
        tables_dir / "improvement_cases.csv",
        improvement_rows,
        [
            "id",
            "question",
            "gold_answer",
            "direct_prediction",
            "self_consistency_prediction",
            "debate_prediction",
            "direct_correct",
            "self_consistency_correct",
            "debate_correct",
            "consensus_reached",
            "stopped_early",
        ]
    )

    write_blog_tables(
        output_path=tables_dir / "blog_tables.md",
        metrics=metrics,
        direct_results=direct_results_flagged,
        sc_results=sc_results_flagged,
        debate_results=debate_results_flagged,
    )

    # Save figures
    make_accuracy_bar(metrics, figures_dir / "accuracy_bar.png")
    make_judge_confidence_hist(debate_results_flagged, figures_dir / "judge_confidence_hist.png")
    make_consensus_bar(debate_results_flagged, figures_dir / "consensus_bar.png")

    return metrics








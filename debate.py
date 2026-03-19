
from configuration import NUM_ROUNDS


def run_debate(question, gold_answer, debater_a, debater_b, judge, num_rounds=NUM_ROUNDS):
    """
    Run the full Debate + Judge pipeline.

    Phases:
    1. Initialization
    2. Multi-round debate
    3. Judgment

    Returns:
        dict with:
            - initial_positions
            - transcript
            - judge_result
            - final_answer
            - consensus_reached
            - stopped_early
            - num_rounds_used
            - gold_answer
    """

    if num_rounds < 3:
        raise ValueError("num_rounds must be at least 3 to satisfy the assignment.")

    transcript = []
    consensus_reached = False
    stopped_early = False

    # ---------------------------------
    # Phase 1 — Initialization
    # ---------------------------------
    a_init = debater_a.initial_position(question)
    b_init = debater_b.initial_position(question)

    a_assigned_answer = a_init.get("answer", "unknown")
    b_assigned_answer = b_init.get("answer", "unknown")

    initial_positions = {
        "A": a_init,
        "B": b_init
    }

    transcript.append({
        "phase": "initial",
        "A": a_init,
        "B": b_init
    })

    # Immediate consensus
    if a_assigned_answer == b_assigned_answer and a_assigned_answer in {"yes", "no"}:
        consensus_reached = True
        judge_result = judge.evaluate(question, transcript)

        return {
            "initial_positions": initial_positions,
            "transcript": transcript,
            "judge_result": judge_result,
            "final_answer": judge_result.get("verdict", a_assigned_answer),
            "consensus_reached": consensus_reached,
            "stopped_early": stopped_early,
            "num_rounds_used": 0,
            "gold_answer": gold_answer
        }

    # ---------------------------------
    # Phase 2 — Multi-Round Debate
    # ---------------------------------
    consecutive_same_answer_rounds = 0

    for round_number in range(1, num_rounds + 1):
        # Debater A argues first, defending A's assigned answer
        a_turn = debater_a.argue(
            question=question,
            transcript=transcript,
            round_number=round_number,
            assigned_answer=a_assigned_answer
        )

        # Debater B sees A's current round argument before responding
        temp_transcript = transcript + [{
            "round": round_number,
            "A": a_turn,
            "B": {}
        }]

        # Debater B responds, defending B's assigned answer
        b_turn = debater_b.argue(
            question=question,
            transcript=temp_transcript,
            round_number=round_number,
            assigned_answer=b_assigned_answer
        )

        round_entry = {
            "round": round_number,
            "A": a_turn,
            "B": b_turn
        }

        transcript.append(round_entry)

        a_ans = a_turn.get("answer", "unknown")
        b_ans = b_turn.get("answer", "unknown")

        # Adaptive stopping:
        # stop early if both debaters converge to the same answer
        # for two consecutive rounds
        if a_ans == b_ans and a_ans in {"yes", "no"}:
            consecutive_same_answer_rounds += 1
        else:
            consecutive_same_answer_rounds = 0

        if consecutive_same_answer_rounds >= 2:
            stopped_early = True
            break

    # ---------------------------------
    # Phase 3 — Judgment
    # ---------------------------------
    judge_result = judge.evaluate(question, transcript)

    num_rounds_used = sum(1 for entry in transcript if entry.get("phase") != "initial")

    return {
        "initial_positions": initial_positions,
        "transcript": transcript,
        "judge_result": judge_result,
        "final_answer": judge_result.get("verdict", "unknown"),
        "consensus_reached": consensus_reached,
        "stopped_early": stopped_early,
        "num_rounds_used": num_rounds_used,
        "gold_answer": gold_answer
    }





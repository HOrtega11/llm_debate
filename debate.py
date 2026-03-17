


def run_debate(question, gold_answer, debater_a, debater_b, judge, num_rounds=3):
    """
    Run the full Debate + Judge pipeline.

    Phases:
    1. Initialization
    2. Multi-round debate
    3. Judgment

    Args:
        question (str): Original question
        gold_answer (str): Ground-truth answer
        debater_a: DebaterA object
        debater_b: DebaterB object
        judge: Judge object
        num_rounds (int): Maximum number of debate rounds (must be >= 3)

    Returns:
        dict with:
            - initial_positions
            - transcript
            - judge_result
            - final_answer
            - consensus_reached
            - stopped_early
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

    initial_positions = {
        "A": a_init,
        "B": b_init
    }

    transcript.append({
        "phase": "initial",
        "A": a_init,
        "B": b_init
    })

    # If both debaters agree immediately, record consensus and skip to judgment
    if a_init.get("answer") == b_init.get("answer") and a_init.get("answer") in {"yes", "no"}:
        consensus_reached = True

        judge_result = judge.evaluate(question, transcript)

        return {
            "initial_positions": initial_positions,
            "transcript": transcript,
            "judge_result": judge_result,
            "final_answer": judge_result.get("verdict", a_init.get("answer", "unknown")),
            "consensus_reached": consensus_reached,
            "stopped_early": stopped_early,
            "gold_answer": gold_answer
        }

    # ---------------------------------
    # Phase 2 — Multi-Round Debate
    # ---------------------------------
    consecutive_same_answer_rounds = 0

    for round_number in range(1, num_rounds + 1):
        # Debater A goes first each round
        a_turn = debater_a.argue(question, transcript, round_number)

        # Debater B responds after seeing all previous context
        # including A's current round argument
        temp_transcript = transcript + [{
            "round": round_number,
            "A": a_turn.get("argument", ""),
            "B": ""
        }]

        b_turn = debater_b.argue(question, temp_transcript, round_number)

        round_entry = {
            "round": round_number,
            "A": a_turn.get("argument", ""),
            "B": b_turn.get("argument", ""),
            "A_answer": a_turn.get("answer", "unknown"),
            "B_answer": b_turn.get("answer", "unknown")
        }

        transcript.append(round_entry)

        # Adaptive stopping:
        # end early if both agents converge to the same answer
        # for two consecutive rounds
        a_ans = a_turn.get("answer", "unknown")
        b_ans = b_turn.get("answer", "unknown")

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

    return {
        "initial_positions": initial_positions,
        "transcript": transcript,
        "judge_result": judge_result,
        "final_answer": judge_result.get("verdict", "unknown"),
        "consensus_reached": consensus_reached,
        "stopped_early": stopped_early,
        "gold_answer": gold_answer
    }







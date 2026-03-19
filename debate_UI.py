

import streamlit as st
from debate import run_debate
from debaterA import DebaterA
from debaterB import DebaterB
from judge import Judge


def render_summary(result):
    st.subheader("Debate Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Consensus Reached", "Yes" if result.get("consensus_reached") else "No")

    with col2:
        st.metric("Stopped Early", "Yes" if result.get("stopped_early") else "No")

    with col3:
        st.metric("Rounds Used", result.get("num_rounds_used", 0))


def render_transcript(transcript):
    st.subheader("Debate Transcript")

    if not transcript:
        st.info("No transcript available.")
        return

    # Initialization
    init_entry = transcript[0] if transcript and transcript[0].get("phase") == "initial" else None

    if init_entry:
        a_init = init_entry.get("A", {})
        b_init = init_entry.get("B", {})

        st.markdown("### Initialization")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Debater A")
            st.markdown(f"**Initial Answer:** {a_init.get('answer', 'unknown').upper()}")
            st.write(a_init.get("reasoning", ""))

        with col2:
            st.markdown("#### Debater B")
            st.markdown(f"**Initial Answer:** {b_init.get('answer', 'unknown').upper()}")
            st.write(b_init.get("reasoning", ""))

    # Debate rounds
    round_entries = [entry for entry in transcript if entry.get("phase") != "initial"]

    for entry in round_entries:
        round_number = entry.get("round", "?")
        a = entry.get("A", {})
        b = entry.get("B", {})

        st.markdown(f"### Round {round_number}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Debater A")
            st.markdown(f"**Assigned / Current Answer:** {a.get('answer', 'unknown').upper()}")
            st.write(a.get("argument", ""))

        with col2:
            st.markdown("#### Debater B")
            st.markdown(f"**Assigned / Current Answer:** {b.get('answer', 'unknown').upper()}")
            st.write(b.get("argument", ""))


def render_judge_result(judge_result):
    st.subheader("Judge Verdict")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Winner", judge_result.get("winner", "unknown"))

    with col2:
        st.metric("Final Answer", judge_result.get("verdict", "unknown").upper())

    with col3:
        st.metric("Confidence", f"{judge_result.get('confidence', 'unknown')} / 5")

    st.markdown("### Analysis")
    st.write(judge_result.get("analysis", ""))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Debater A")
        st.markdown(f"**Strongest point:** {judge_result.get('strongest_A', '')}")
        st.markdown(f"**Weakest point:** {judge_result.get('weakest_A', '')}")

    with col2:
        st.markdown("### Debater B")
        st.markdown(f"**Strongest point:** {judge_result.get('strongest_B', '')}")
        st.markdown(f"**Weakest point:** {judge_result.get('weakest_B', '')}")


st.set_page_config(page_title="LLM Debate System", layout="wide")

st.title("LLM Debate System")
st.write("Enter a yes/no question and watch two debaters argue before a judge gives the final verdict.")

question = st.text_input("Enter a question")

if st.button("Run Debate"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running debate..."):
            debater_a = DebaterA()
            debater_b = DebaterB()
            judge = Judge()

            result = run_debate(
                question=question,
                gold_answer="unknown",
                debater_a=debater_a,
                debater_b=debater_b,
                judge=judge
            )

        render_summary(result)
        render_transcript(result["transcript"])
        render_judge_result(result["judge_result"])

        with st.expander("Show raw transcript JSON"):
            st.json(result["transcript"])

        with st.expander("Show raw judge JSON"):
            st.json(result["judge_result"])













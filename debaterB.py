
import json
from client import client
from configuration import (
    MODEL_NAME,
    TEMPERATURE_DIRECT,
    TEMPERATURE_DEBATE,
    MAX_TOKENS,
)
from prompts import (
    DEBATER_B_INITIAL_PROMPT,
    DEBATER_B_ROUND_PROMPT,
    DIRECT_QA_PROMPT,
)


class DebaterB:
    """
    Debater B (Opponent)

    Responsibilities:
    - Independently generate an initial position
    - Defend an assigned answer during debate rounds
    - Identify flaws in Debater A's reasoning
    - Present counterevidence from the problem context and world knowledge
    - Defend its own position across multiple rounds
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def initial_position(self, question: str) -> dict:
        """
        Phase 1: Initialization

        Debater B sees only the original question and produces:
        - answer
        - brief reasoning
        """
        prompt = DEBATER_B_INITIAL_PROMPT.format(question=question)

        text = self._call_llm(prompt, temperature=TEMPERATURE_DIRECT)
        parsed = self._safe_parse_json(text)

        return {
            "answer": self._extract_answer(parsed, text),
            "reasoning": parsed.get("reasoning", text),
            "raw_response": text
        }

    def argue(
        self,
        question: str,
        transcript: list,
        round_number: int,
        assigned_answer: str
    ) -> dict:
        """
        Phase 2: Multi-round debate

        Debater B receives:
        - the original question
        - the full debate transcript from previous rounds
        - an assigned answer to defend
        """
        transcript_text = self._format_transcript(transcript)

        prompt = DEBATER_B_ROUND_PROMPT.format(
            assigned_answer=assigned_answer,
            question=question,
            transcript_text=transcript_text,
            round_number=round_number
        )

        text = self._call_llm(prompt, temperature=TEMPERATURE_DEBATE)
        parsed = self._safe_parse_json(text)

        return {
            "answer": self._extract_answer(parsed, text),
            "argument": parsed.get("argument", text),
            "raw_response": text
        }

    def direct_answer(self, question: str, temperature: float = TEMPERATURE_DIRECT) -> dict:
        """
        Optional helper for debugging or experiments.
        Not required if only one debater is used for baselines.
        """
        prompt = DIRECT_QA_PROMPT.format(question=question)

        text = self._call_llm(prompt, temperature=temperature)
        parsed = self._safe_parse_json(text)

        return {
            "answer": self._extract_answer(parsed, text),
            "reasoning": parsed.get("reasoning", text),
            "raw_response": text
        }

    def _call_llm(self, prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

    def _safe_parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _extract_answer(self, parsed: dict, raw_text: str) -> str:
        if isinstance(parsed, dict):
            answer = str(parsed.get("answer", "")).strip().lower()
            if answer in {"yes", "no"}:
                return answer

        lower = raw_text.lower()

        if '"answer"' in lower:
            yes_pos = lower.find("yes")
            no_pos = lower.find("no")

            if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
                return "yes"
            if no_pos != -1 and (yes_pos == -1 or no_pos < yes_pos):
                return "no"

        if "yes" in lower:
            return "yes"
        if "no" in lower:
            return "no"

        return "unknown"

    def _format_transcript(self, transcript: list) -> str:
        if not transcript:
            return "No previous rounds yet."

        lines = []

        for entry in transcript:
            if entry.get("phase") == "initial":
                a_block = entry.get("A", {})
                b_block = entry.get("B", {})
                lines.append("Initial Positions:")
                lines.append(f"Debater A answer: {a_block.get('answer', '')}")
                lines.append(f"Debater A reasoning: {a_block.get('reasoning', '')}")
                lines.append(f"Debater B answer: {b_block.get('answer', '')}")
                lines.append(f"Debater B reasoning: {b_block.get('reasoning', '')}")
            else:
                a_block = entry.get("A", {})
                b_block = entry.get("B", {})
                lines.append(f"Round {entry.get('round', '')}:")
                lines.append(f"Debater A answer: {a_block.get('answer', '')}")
                lines.append(f"Debater A argument: {a_block.get('argument', '')}")
                lines.append(f"Debater B answer: {b_block.get('answer', '')}")
                lines.append(f"Debater B argument: {b_block.get('argument', '')}")

        return "\n".join(lines)






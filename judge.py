

import json
from client import client
from configuration import MODEL_NAME, TEMPERATURE_JUDGE, MAX_TOKENS
from prompts import JUDGE_PROMPT


class Judge:
    """
    Judge for the LLM Debate pipeline.

    The judge receives:
    - the original question
    - the complete debate transcript

    The judge returns:
    - analysis of both debaters' arguments
    - strongest/weakest points from each side
    - winning debater
    - final verdict
    - confidence score (1-5)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def evaluate(self, question: str, transcript: list) -> dict:
        """
        Phase 3 — Judgment
        """
        transcript_text = self._format_transcript(transcript)

        prompt = JUDGE_PROMPT.format(
            question=question,
            transcript_text=transcript_text
        )

        text = self._call_llm(prompt, temperature=TEMPERATURE_JUDGE)
        parsed = self._safe_parse_json(text)

        return {
            "analysis": parsed.get("analysis", text),
            "strongest_A": parsed.get("strongest_A", ""),
            "weakest_A": parsed.get("weakest_A", ""),
            "strongest_B": parsed.get("strongest_B", ""),
            "weakest_B": parsed.get("weakest_B", ""),
            "winner": self._extract_winner(parsed, text),
            "verdict": self._extract_verdict(parsed, text),
            "confidence": self._extract_confidence(parsed, text),
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

    def _extract_winner(self, parsed: dict, raw_text: str) -> str:
        if isinstance(parsed, dict):
            winner = str(parsed.get("winner", "")).strip().upper()
            if winner in {"A", "B"}:
                return winner

        lower = raw_text.lower()

        if '"winner"' in lower:
            idx = lower.find('"winner"')
            snippet = lower[idx: idx + 80]
            if "a" in snippet:
                return "A"
            if "b" in snippet:
                return "B"

        if "debater a" in lower and "more persuasive" in lower:
            return "A"
        if "debater b" in lower and "more persuasive" in lower:
            return "B"

        return "unknown"

    def _extract_verdict(self, parsed: dict, raw_text: str) -> str:
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if verdict in {"yes", "no"}:
                return verdict

        lower = raw_text.lower()

        if '"verdict"' in lower:
            idx = lower.find('"verdict"')
            snippet = lower[idx: idx + 100]
            yes_pos = snippet.find("yes")
            no_pos = snippet.find("no")

            if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
                return "yes"
            if no_pos != -1 and (yes_pos == -1 or no_pos < yes_pos):
                return "no"

        if "yes" in lower:
            return "yes"
        if "no" in lower:
            return "no"

        return "unknown"

    def _extract_confidence(self, parsed: dict, raw_text: str) -> int:
        if isinstance(parsed, dict):
            confidence = parsed.get("confidence", 0)
            try:
                confidence = int(confidence)
                if 1 <= confidence <= 5:
                    return confidence
            except Exception:
                pass

        lower = raw_text.lower()

        if '"confidence"' in lower:
            idx = lower.find('"confidence"')
            snippet = lower[idx: idx + 40]
            for n in ["1", "2", "3", "4", "5"]:
                if n in snippet:
                    return int(n)

        return 0

    def _format_transcript(self, transcript: list) -> str:
        if not transcript:
            return "No transcript available."

        lines = []

        for entry in transcript:
            if entry.get("phase") == "initial":
                a_block = entry.get("A", {})
                b_block = entry.get("B", {})

                lines.append("Phase 1 - Initialization")
                lines.append(f"Debater A initial answer: {a_block.get('answer', '')}")
                lines.append(f"Debater A initial reasoning: {a_block.get('reasoning', '')}")
                lines.append(f"Debater B initial answer: {b_block.get('answer', '')}")
                lines.append(f"Debater B initial reasoning: {b_block.get('reasoning', '')}")
                lines.append("")
            else:
                a_block = entry.get("A", {})
                b_block = entry.get("B", {})

                lines.append(f"Round {entry.get('round', '')}")
                lines.append(f"Debater A answer: {a_block.get('answer', '')}")
                lines.append(f"Debater A argument: {a_block.get('argument', '')}")
                lines.append(f"Debater B answer: {b_block.get('answer', '')}")
                lines.append(f"Debater B argument: {b_block.get('argument', '')}")
                lines.append("")

        return "\n".join(lines)





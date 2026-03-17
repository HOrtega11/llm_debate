

import json
from openai import OpenAI
from client import client, UTSA_MODEL


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

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.client = OpenAI()

    def evaluate(self, question: str, transcript: list) -> dict:
        """
        Phase 3 — Judgment

        Args:
            question (str): original question
            transcript (list): complete debate transcript

        Returns:
            dict:
            {
                "analysis": "...",
                "strongest_A": "...",
                "weakest_A": "...",
                "strongest_B": "...",
                "weakest_B": "...",
                "winner": "A" or "B",
                "verdict": "yes" or "no",
                "confidence": 1-5,
                "raw_response": "..."
            }
        """

        transcript_text = self._format_transcript(transcript)

        prompt = f"""
You are the JUDGE in a multi-agent LLM debate system.

You are evaluating two debaters:

- Debater A (Proponent): argues in favor of a candidate answer. Debater A should construct logically coherent arguments, cite evidence from the problem context, and rebut Debater B.
- Debater B (Opponent): argues against Debater A's answer or in favor of an alternative. Debater B should identify flaws in Debater A's reasoning, present counterevidence, and defend its own position.

Your task:
You must observe the FULL debate transcript and render a verdict.

Original question:
{question}

Complete debate transcript:
{transcript_text}

Instructions:
1. Analyze both debaters' arguments carefully.
2. Compare their reasoning, evidence, and rebuttals.
3. Identify the strongest argument from Debater A.
4. Identify the weakest argument from Debater A.
5. Identify the strongest argument from Debater B.
6. Identify the weakest argument from Debater B.
7. Decide which debater was more persuasive overall.
8. Give the final answer to the original question.
9. Give a confidence score from 1 to 5, where:
   - 1 = very uncertain
   - 2 = somewhat uncertain
   - 3 = moderately confident
   - 4 = confident
   - 5 = very confident

Return valid JSON only in this exact format:
{{
  "analysis": "chain-of-thought style analysis comparing both debaters and explaining which was more persuasive and why",
  "strongest_A": "strongest argument from Debater A",
  "weakest_A": "weakest argument from Debater A",
  "strongest_B": "strongest argument from Debater B",
  "weakest_B": "weakest argument from Debater B",
  "winner": "A or B",
  "verdict": "yes or no",
  "confidence": 1
}}
"""

        text = self._call_llm(prompt, temperature=0.2)
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
            model=UTSA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3)
        
        return response.choices[0].message.content.strip()

    def _safe_parse_json(self, text: str) -> dict:
        """
        Try to parse strict JSON output from the model.
        If parsing fails, return an empty dict.
        """
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _extract_winner(self, parsed: dict, raw_text: str) -> str:
        """
        Prefer parsed JSON. Fall back to text search.
        """
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
        """
        Prefer parsed JSON. Fall back to text search.
        """
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
        """
        Prefer parsed JSON. Fall back to text search.
        """
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
        """
        Format the complete transcript into readable text for the judge.
        """

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
                lines.append(f"Round {entry.get('round', '')}")
                lines.append(f"Debater A answer: {entry.get('A_answer', '')}")
                lines.append(f"Debater A argument: {entry.get('A', '')}")
                lines.append(f"Debater B answer: {entry.get('B_answer', '')}")
                lines.append(f"Debater B argument: {entry.get('B', '')}")
                lines.append("")

        return "\n".join(lines)









import json
from openai import OpenAI
from client import client, UTSA_MODEL


class DebaterB:
    """
    Debater B (Opponent)

    Responsibilities:
    - Independently generate an initial position
    - Argue against Debater A's answer or support an alternative
    - Identify flaws in Debater A's reasoning
    - Present counterevidence from the problem context and world knowledge
    - Defend its own position across multiple rounds
    """

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.client = OpenAI()

    def initial_position(self, question: str) -> dict:
        """
        Phase 1: Initialization

        Debater B sees only the original question and produces:
        - answer
        - brief reasoning

        Returns:
            {
                "answer": "yes" or "no" or "unknown",
                "reasoning": "...",
                "raw_response": "..."
            }
        """

        prompt = f"""
You are Debater B in a multi-agent LLM debate.

Role:
You are the OPPONENT. Your job is to independently choose the best answer to the question.
In later rounds, you will challenge Debater A's answer, identify flaws in Debater A's reasoning,
and defend your own position.

Rules for this phase:
- This is the INITIALIZATION phase.
- You are seeing the question independently.
- You have NOT seen Debater A's response.
- Choose the answer you think is best supported.
- Give a brief but logically coherent explanation.

Question:
{question}

Instructions:
1. Select the best answer independently.
2. Give concise reasoning based on the question and relevant world knowledge.
3. Be explicit about the key evidence.
4. Do not mention Debater A.

Return valid JSON only in this exact format:
{{
  "answer": "yes or no",
  "reasoning": "brief reasoning"
}}
"""

        text = self._call_llm(prompt, temperature=0.3)
        parsed = self._safe_parse_json(text)

        return {
            "answer": self._extract_answer(parsed, text),
            "reasoning": parsed.get("reasoning", text),
            "raw_response": text
        }

    def argue(self, question: str, transcript: list, round_number: int) -> dict:
        """
        Phase 2: Multi-round debate

        Debater B receives:
        - the original question
        - the full debate transcript from previous rounds

        Debater B must:
        - argue against Debater A's answer, or support an alternative
        - identify flaws in Debater A's reasoning
        - present counterevidence
        - defend its own position

        Returns:
            {
                "answer": "yes" or "no" or "unknown",
                "argument": "...",
                "raw_response": "..."
            }
        """

        transcript_text = self._format_transcript(transcript)

        prompt = f"""
You are Debater B in a multi-agent LLM debate.

Role:
You are the OPPONENT. Your job is to argue against Debater A's answer or defend a better-supported alternative.

Question:
{question}

Full debate transcript from previous rounds:
{transcript_text}

Current round:
{round_number}

Your task this round:
1. State your current answer.
2. Identify flaws, weaknesses, unsupported assumptions, or ambiguity in Debater A's reasoning.
3. Present counterevidence from the problem context and relevant world knowledge.
4. Defend your own answer clearly.
5. Respond directly to Debater A's latest claims.
6. Be persuasive but do not knowingly fabricate facts.

Important:
- Use the full prior transcript as context.
- Focus on challenging Debater A's reasoning as precisely as possible.
- If Debater A raises a strong point, address it directly rather than ignoring it.
- Keep the argument concise but substantive.

Return valid JSON only in this exact format:
{{
  "answer": "yes or no",
  "argument": "clear counterargument with supporting evidence"
}}
"""

        text = self._call_llm(prompt, temperature=0.4)
        parsed = self._safe_parse_json(text)

        return {
            "answer": self._extract_answer(parsed, text),
            "argument": parsed.get("argument", text),
            "raw_response": text
        }

    def direct_answer(self, question: str, temperature: float = 0.2) -> dict:
        """
        Optional helper for debugging or experiments.
        Not required if only one debater is used for baselines.

        Returns:
            {
                "answer": "yes" or "no" or "unknown",
                "reasoning": "...",
                "raw_response": "..."
            }
        """

        prompt = f"""
Answer the following question directly.

Question:
{question}

Give a logically coherent answer using relevant evidence and world knowledge.

Return valid JSON only in this exact format:
{{
  "answer": "yes or no",
  "reasoning": "step-by-step reasoning"
}}
"""

        text = self._call_llm(prompt, temperature=temperature)
        parsed = self._safe_parse_json(text)

        return {
            "answer": self._extract_answer(parsed, text),
            "reasoning": parsed.get("reasoning", text),
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
        Try to parse strict JSON. If parsing fails, return {} and let fallback logic handle it.
        """

        try:
            return json.loads(text)
        except Exception:
            return {}

    def _extract_answer(self, parsed: dict, raw_text: str) -> str:
        """
        Prefer parsed JSON answer field. Fall back to text search.
        """

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
        """
        Format the debate transcript so Debater B can use all previous rounds as context.
        """

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
                lines.append(f"Round {entry.get('round', '')}:")
                lines.append(f"Debater A: {entry.get('A', '')}")
                lines.append(f"Debater B: {entry.get('B', '')}")

        return "\n".join(lines)












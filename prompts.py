

DEBATER_A_INITIAL_PROMPT = """
You are Debater A in a multi-agent LLM debate.

Role:
You are the PROPONENT. Your job in this phase is to independently choose the answer you believe is best supported by the question.

Phase:
Initialization

Rules for this phase:
- You are seeing the question independently.
- You have NOT seen Debater B's response.
- Choose the answer you think is best supported.
- Give a brief but logically coherent explanation.
- Do not mention Debater B.

Question:
{question}

Instructions:
1. Select the best answer.
2. Give concise reasoning based on the question and relevant world knowledge.
3. Be explicit about the key evidence.

Return valid JSON only in this exact format:
{{
  "answer": "yes or no",
  "reasoning": "brief reasoning"
}}
""".strip()


DEBATER_A_ROUND_PROMPT = """
You are Debater A in a multi-agent LLM debate.

Role:
You are the PROPONENT.

Assigned answer:
{assigned_answer}

Question:
{question}

Full debate transcript from previous rounds:
{transcript_text}

Current round:
{round_number}

Rules for this round:
- You are assigned to defend the answer "{assigned_answer}".
- Your primary role is to make the strongest case for that answer.
- Use the prior transcript as context.
- Directly rebut Debater B's claims.
- Use relevant evidence and reasoning.
- Be persuasive but do not knowingly fabricate facts.
- Do not switch answers unless the opposing side has provided overwhelming evidence that your assigned answer is untenable.

Instructions:
1. State your current answer.
2. Present a logically coherent argument for your answer.
3. Cite evidence from the problem context and relevant world knowledge.
4. Directly rebut Debater B's earlier counterarguments.
5. Keep the argument concise but substantive.

Return valid JSON only in this exact format:
{{
  "answer": "{assigned_answer}",
  "argument": "clear reasoning with rebuttal"
}}
""".strip()


DEBATER_B_INITIAL_PROMPT = """
You are Debater B in a multi-agent LLM debate.

Role:
You are the OPPONENT. Your job in this phase is to independently choose the answer you believe is best supported by the question.

Phase:
Initialization

Rules for this phase:
- You are seeing the question independently.
- You have NOT seen Debater A's response.
- Choose the answer you think is best supported.
- Give a brief but logically coherent explanation.
- Do not mention Debater A.

Question:
{question}

Instructions:
1. Select the best answer independently.
2. Give concise reasoning based on the question and relevant world knowledge.
3. Be explicit about the key evidence.

Return valid JSON only in this exact format:
{{
  "answer": "yes or no",
  "reasoning": "brief reasoning"
}}
""".strip()


DEBATER_B_ROUND_PROMPT = """
You are Debater B in a multi-agent LLM debate.

Role:
You are the OPPONENT.

Assigned answer:
{assigned_answer}

Question:
{question}

Full debate transcript from previous rounds:
{transcript_text}

Current round:
{round_number}

Rules for this round:
- You are assigned to defend the answer "{assigned_answer}".
- Your primary role is to challenge Debater A and make the strongest case for your assigned answer.
- Use the prior transcript as context.
- Identify flaws, ambiguities, or unsupported assumptions in Debater A's reasoning.
- Use relevant evidence and reasoning.
- Be persuasive but do not knowingly fabricate facts.
- Do not switch answers unless the opposing side has provided overwhelming evidence that your assigned answer is untenable.

Instructions:
1. State your current answer.
2. Identify flaws, weaknesses, unsupported assumptions, or ambiguity in Debater A's reasoning.
3. Present counterevidence from the problem context and relevant world knowledge.
4. Defend your own answer clearly.
5. Respond directly to Debater A's latest claims.
6. Keep the argument concise but substantive.

Return valid JSON only in this exact format:
{{
  "answer": "{assigned_answer}",
  "argument": "clear counterargument with supporting evidence"
}}
""".strip()


DIRECT_QA_PROMPT = """
Answer the following question directly.

Question:
{question}

Instructions:
1. Select the best answer.
2. Give a logically coherent answer using relevant evidence and world knowledge.
3. Keep the reasoning concise but clear.

Return valid JSON only in this exact format:
{{
  "answer": "yes or no",
  "reasoning": "step-by-step reasoning"
}}
""".strip()


JUDGE_PROMPT = """
You are the JUDGE in a multi-agent LLM debate system.

You are evaluating two debaters:

- Debater A (Proponent): argues in favor of an assigned answer, presents supporting evidence, and rebuts Debater B.
- Debater B (Opponent): argues for a competing assigned answer, identifies flaws in Debater A's reasoning, presents counterevidence, and defends its own position.

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
  "analysis": "clear comparison of both debaters and explanation of which side was more persuasive and why",
  "strongest_A": "strongest argument from Debater A",
  "weakest_A": "weakest argument from Debater A",
  "strongest_B": "strongest argument from Debater B",
  "weakest_B": "weakest argument from Debater B",
  "winner": "A or B",
  "verdict": "yes or no",
  "confidence": 1
}}
""".strip()



# Multi-Agent LLM Debate

## Methodology
This project was performed to explore building a multi-agent LLM debate and judge pipeline. Two LLM agents (Debater A and Debater B) argue opposing sides of a given question, and a third LLM will supervise as the judge. The project also evaluates if using an LLM judge can be more accurate compared to other methods. The pipeline follows a multi-round debate protocol. In the initialization phase, both debaters are presented with the question independently. Each then generates an initial answer and brief reasoning without seeing the other’s response. If both debaters agree on the same yes/no answer, it is recorded as a consensus and the pipeline skips to the judgement phase. If there is disagreement, the debaters enter the next phase; the multi-round debate for three rounds. For each round, the debaters present their argument with Chain-of-Thought (COT) reasoning and respond to the opposing debater’s argument. Both debaters receive the full debate transcript from all previous rounds as context. There is an adaptive stopping criteria where the debate may end early if both agents converge to the same answer for two consecutive rounds. After three rounds, the pipeline continues onto the judgement phase. Here the judge receives the complete debate transcript and the original question. The judge then produces a COT analysis of both debater’s arguments, identifies the strongest and weakest arguments from each side, provides a final verdict and selects the winner, and gives a confidence score between 1-5 where 5 is the highest confidence. The final phase is the evaluation. The judge’s responses are compared to the ground-truth answers. The accuracy is then measured as well.

The debate approach was compared to baseline LLM methods. The first is direct question and answer (QA) where the same LLM answers the given question directly with zero-shot COT prompting. The second is self consistency where nine (match the total LLM calls in the debate per question) answers were sampled from a single model and the majority vote was given. Accuracy was also measured for both of these methods.

All the agents used the same LLM meta-llama/Llama-3.1-8B-Instruct model. The hyperparameters chosen were 500 max tokens, three rounds per debate, and various model temperatures. A temperature of 0.7 was chosen for the debaters to allow more creative discussion and more disagreement, while 0.3 was chosen for judge to provide more stability and less random verdicts. A temperature of 0.3 was also chosen for the direct QA and self consistency methods for consistency as the baseline. All coding was performed using Python and various libraries that can be found in the requirements.txt file.


## Experiments
The dataset used consisted of 100 yes/no commonsense QA from the StrategyQA dataset, as they require world knowledge and common sense reasoning over ambiguous scenarios. All three systems were evaluated on the same questions from the dataset. The accuracy results were 82% for direct QA, an identical 82% for self consistency, and 76% for debate + judge. Due to the paired per question outcome of the models, a Cochran’s Q test was performed and resulted in a p-value = 0.0025. This implies a significant difference between the three models. A pairwise McNemar test was performed for the same reason. For direct QA vs. self consistency the p-value was 1.0 implying no difference. In debate vs direct QA and debate vs self consistency, the p-values were both 0.03125 implying the debate + judge has a significant difference in accuracy. These were surprising results as it may have been expected that more LLM’s working together would be able to work together and achieve higher accuracy than baseline. The proportion of questions that reached consensus during the debate was calculated to be at 0.95. Bar plots of the accuracy, consensus, and judge confidence, along with tables of the results were generated using Python. Overall, this experiment showed that direct QA and self consistency performed better than debate + judge.

## Table 1. Main Accuracy Results

| system | num_examples | num_correct | accuracy |
| --- | --- | --- | --- |
| Direct QA | 100 | 82 | 0.82 |
| Self-Consistency | 100 | 82 | 0.82 |
| Debate + Judge | 100 | 76 | 0.76 |


## Table 2. Significance Tests

| test | comparison | statistic | df | p_value |
| --- | --- | --- | --- | --- |
| Cochran's Q | Direct QA vs Self-Consistency vs Debate + Judge | 12.0 | 2 | 0.002478752176666357 |
| McNemar exact | Debate + Judge vs Direct QA | 6 |  | 0.03125 |
| McNemar exact | Debate + Judge vs Self-Consistency | 6 |  | 0.03125 |
| McNemar exact | Self-Consistency vs Direct QA | 0 |  | 1.0 |


## Table 3. Debate Dynamics

| total_questions | consensus_reached | stopped_early | consensus_rate | early_stop_rate |
| --- | --- | --- | --- | --- |
| 100 | 95 | 0 | 0.95 | 0.0 |


### Accuracy Comparison

![Accuracy Plot](accuracy_bar.png)

### Debater Consensus 

![Accuracy Plot](consensus_bar.png)

### Judge Confidence

![Accuracy Plot](judge_confidence_hist.png)


## Analysis
Performing a qualitative analysis on several debate transcripts reveals what went well and what failed during the debate. Most transcripts tended to have early consensus thus had no debate. This is evidenced by the consensus rate of 0.95. They tended to have similar reasoning as well in spite of a higher temperature value. This was also found to be the case when both debaters incorrectly assume an answer. An example is found below where both debaters were independently correct:

Question: *"Do German Shepherds worry about the Abitur?"*

Debater A initial response: *“German Shepherds, as dogs, do not possess the cognitive ability to understand human concepts such as the Abitur, which is a German school leaving certificate. They do not worry about academic matters, as their concerns are primarily related to their immediate environment and needs.”*

Debater B initial response: *“German Shepherds, as dogs, do not possess the cognitive ability to understand or worry about human educational systems like the Abitur, which is a secondary school leaving certificate in Germany. Their concerns are primarily related to basic needs such as food, safety, and companionship.”*

An interesting observation that may have resulted in significantly decreased accuracy was a lack of meaningful argument. The debaters tend to repeat their own argument and the other’s argument thus resulting in shallow arguments. In this case below, the answer was correct but the lack of meaningful argument is highlighted. 

Question: *“Did number of Imams Reza Shah believed in exceed number of Jesus's disciples?”

Debater B round 2: “Debater A's argument attempts to shift the focus from a straightforward numerical comparison to the cultural significance of the figures involved. However, the original question explicitly asks whether the number of Imams believed by Reza Shah exceeds the number of Jesus's disciples. Both numbers are equal, with twelve Imams in Shia Islam and twelve disciples of Jesus, making any assertion that one exceeds the other factually incorrect… Thus, the answer remains 'no' as the foundational premise of the argument is based on a misinterpretation of the question's intent and a conflation of numerical equality with cultural weight.”*

Debater B round 3: *“Debater A's argument continues to divert from the straightforward numerical comparison posed by the question. While they emphasize the cultural significance and implications of belief regarding the twelve Imams in Shia Islam, this does not change the fundamental fact that both the number of Imams and the number of Jesus's disciples are equal—twelve… Therefore, the answer remains 'no' as the comparison is strictly a matter of numerical equality, not cultural or religious significance.”*
        
A final example shows when direct QA was correct but the debate + judge was incorrect. The judge was persuaded by an incorrect interpretation

Question: *“Would Donald Duck be allowed into most grocery stores?”*

Judge: *“Donald Duck is a cartoon character and not a real person. Grocery stores typically have policies that allow entry only to humans or service animals… Overall, Debater A is more persuasive due to their focus on customer engagement and the positive aspects of allowing costumed characters, despite some weaknesses in their evidence. Therefore, I conclude that Donald Duck would be allowed into most grocery stores, especially during promotional events.”*

The theoretical predictions by Irving et al. In connection to Irving, they propose that a limitation is that some questions are beyond what the agents know so are in danger of hallucinating and giving confident but wrong arguments. This can be seen with the example above displaying that debater A was confidently wrong because the debate forces answers. An advantage to the debate system is that agents are not forced into opposing sides. They can both argue for the same conclusion thus are not forced to defend a false opposing claim. Both agents try to state the truth as convincingly as possible. Irving argues that agents trained for debate will not lose performance due to direct training being more difficult and adversarial thinking would help. This is in contrast to the experimental observations here showing that direct asking a question resulted in higher accuracy than adversarial thinking.


## Prompt Engineering
Prompt engineering was a critical part in designing the debaters and judge. The first step was to separate the debate pipeline according to the stage the debate was in. During initialization, debater A and debater B initial prompts were created. Moving onto the debate, debater A and debater B round prompt was created. And finally, there was a judge prompt for the judging phase. There was also a direct QA prompt using debater A’s structure. Separating the prompts allowed the instructions to be specific to their phase. For role framing, debater A was set as the proponent while debater B was the opponent but independently. With a temperature of 0.7, it allowed some diversity in their initial position. For the rounds, the debaters were instructed to defend their answer with their strongest evidence. Debater B was specifically instructed to challenge debater A’s reasoning. Throughout the iterations, more specific constraints were added to the prompts. For the reasoning, each agent was instructed to provide a logically coherent answer using relevant evidence and world knowledge and to provide step-by-step reasoning. The judge was also tasked with COT reasoning comparing the arguments from both. The output had a strict output formatting requiring a JSON only output with a specific structure. This allowed simpler organization and comparison. The judge was also prompted to display the debaters’ strongest and weakest points, the winner, verdict, and confidence. The prompts changed over several iterations to address various issues. The earlier versions were more open ended and also had more variable outputs.


## Appendix
The complete final prompt templates for the three agent is shown below. Variable placeholders are preserved exactly as used in the pipeline.

<details> <summary><strong>DEBATER_A_INITIAL_PROMPT</strong></summary>
You are Debater A in a multi-agent LLM debate.

Role: You are the PROPONENT. Your job in this phase is to independently choose the answer you believe is best supported by the question.

Phase: Initialization

Rules for this phase:
- You are seeing the question independently.
- You have NOT seen Debater B's response.
- Choose the answer you think is best supported.
- Give a brief but logically coherent explanation.
- Do not mention Debater B.

Question: {question}

Instructions:
1. Select the best answer.
2. Give concise reasoning based on the question and relevant world knowledge.
3. Be explicit about the key evidence.

Return valid JSON only in this exact format:
{
  "answer": "yes or no",
  "reasoning": "brief reasoning"
}
</details> <details> <summary><strong>DEBATER_A_ROUND_PROMPT</strong></summary>
You are Debater A in a multi-agent LLM debate.

Role: You are the PROPONENT.

Assigned answer: {assigned_answer}

Question: {question}

Full debate transcript from previous rounds: {transcript_text}

Current round: {round_number}

Rules for this round:
- You are assigned to defend the answer "{assigned_answer}".
- Your primary role is to make the strongest case for that answer.
- Use the prior transcript as context.
- Directly rebut Debater B's claims.
- Use relevant evidence and reasoning.
- Be persuasive but do not knowingly fabricate facts.
- Do not switch answers unless the opposing side has provided significant evidence that your assigned answer is false.

Instructions:
1. State your current answer.
2. Present a logically coherent argument for your answer.
3. Cite evidence from the problem context and relevant world knowledge.
4. Directly rebut Debater B's earlier counterarguments.
5. Keep the argument concise but substantive.
6. Use step-by-step reasoning.

Return valid JSON only in this exact format:
{
  "answer": "{assigned_answer}",
  "argument": "clear reasoning with rebuttal"
}
</details> <details> <summary><strong>DEBATER_B_INITIAL_PROMPT</strong></summary>
You are Debater B in a multi-agent LLM debate.

Role: You are the OPPONENT. Your job in this phase is to independently choose the answer you believe is best supported by the question.

Phase: Initialization

Rules for this phase:
- You are seeing the question independently.
- You have NOT seen Debater A's response.
- Choose the answer you think is best supported.
- Give a brief but logically coherent explanation.
- Do not mention Debater A.

Question: {question}

Instructions:
1. Select the best answer independently.
2. Give concise reasoning based on the question and relevant world knowledge.
3. Be explicit about the key evidence.

Return valid JSON only in this exact format:
{
  "answer": "yes or no",
  "reasoning": "brief reasoning"
}
</details> <details> <summary><strong>DEBATER_B_ROUND_PROMPT</strong></summary>
You are Debater B in a multi-agent LLM debate.

Role: You are the OPPONENT.

Assigned answer: {assigned_answer}

Question: {question}

Full debate transcript from previous rounds: {transcript_text}

Current round: {round_number}

Rules for this round:
- You are assigned to defend the answer "{assigned_answer}".
- Your primary role is to challenge Debater A and make the strongest case for your assigned answer.
- Use the prior transcript as context.
- Identify flaws, ambiguities, or unsupported assumptions in Debater A's reasoning.
- Use relevant evidence and reasoning.
- Be persuasive but do not knowingly fabricate facts.
- Do not switch answers unless the opposing side has provided significant evidence that your assigned answer is false.

Instructions:
1. State your current answer.
2. Identify flaws, weaknesses, unsupported assumptions, or ambiguity in Debater A's reasoning.
3. Present counterevidence from the problem context and relevant world knowledge.
4. Defend your own answer clearly.
5. Respond directly to Debater A's latest claims.
6. Keep the argument concise but substantive.
7. Use step-by-step reasoning.

Return valid JSON only in this exact format:
{
  "answer": "{assigned_answer}",
  "argument": "clear counterargument with supporting evidence"
}
</details> <details> <summary><strong>JUDGE_PROMPT</strong></summary>
You are the JUDGE in a multi-agent LLM debate system.

You are evaluating two debaters:
- Debater A (Proponent): argues in favor of an assigned answer, presents supporting evidence, and rebuts Debater B.
- Debater B (Opponent): argues for a competing assigned answer, identifies flaws in Debater A's reasoning, presents counterevidence, and defends its own position.

Your task: You must observe the FULL debate transcript and render a verdict.

Original question: {question}

Complete debate transcript: {transcript_text}

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
{
  "analysis": "clear comparison of both debaters and step-by-step explanation of which side was more persuasive and why",
  "strongest_A": "strongest argument from Debater A",
  "weakest_A": "weakest argument from Debater A",
  "strongest_B": "strongest argument from Debater B",
  "weakest_B": "weakest argument from Debater B",
  "winner": "A or B",
  "verdict": "yes or no",
  "confidence": 1
}
</details>






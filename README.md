# llm_debate
Building Adversarial Multi-Agent Reasoning Systems
 
Overview

This project builds and evaluates a multi-agent debate + judge pipeline where two LLM agents argue opposing sides of a question and a third LLM serves as the judge over multiple rounds. the project also compares three systems. First, a direct QA where a single LLM answers the question directly with zero-shot Chain of Thought (COT) prompting without debating. Second, self consistency where N answers are sampled (N matches the total LLM calls in the debate) from a single model and take the majority vote. And third, the debate as stated above

This project also contains the modular code along with the configuration of hyperparameters, prompt templates, experiment logging, statistical evaluation, and a Streamlit web UI.

main.py: runs all experiments

debate.py: performs initialization, debate rounds, and judgement

debaterA.py: debater A code

debaterB.py: debater B code

judge.py: judge code

prompts.py: all prompt templates for debaters and judge

configuration.py: model names and hyperparamenters

client.py: OnpenAI compatible API client

evaluation.py: evaluation and statistical testing 

debate_UI.py: Streamlit web interface

questions.json: questions tested

debate_results.json: output results

Dependencies are found in requirements.txt

accuracy_bar.png: accuracy results bar graph

consensus_bar.png: consensus results bar graph

judge_confidence_hist: judge confidence results histrogram




Setup instructions
1) clone the repository
2) create a virtual environment
3) install dependencies
4) set API key
5) modify experiment setting in configuration.py if desired
6) to run the full experiment, run "python main.py" on the terminal
7) to run the Streamlit interface for interactive use, run "streamlit run debate_UI.py" on the terminal







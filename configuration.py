
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

TEMPERATURE_DIRECT = 0.3
TEMPERATURE_DEBATE = 0.7
TEMPERATURE_JUDGE = 0.3

MAX_TOKENS = 500

NUM_QUESTIONS = 100
NUM_ROUNDS = 3
SELF_CONSISTENCY_SAMPLES = 2 + (2 * NUM_ROUNDS) + 1 #DebaterA initial + DebatorB initial + 2 calls per round + judge















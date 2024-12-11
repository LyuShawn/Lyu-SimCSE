import random

knowledge_prompt_list = [
    "The phrase \"{sentence}\" may imply \"{knowledge}\", suggesting that [MASK].",
    "Interpreting \"{sentence}\" reveals a connection to \"{knowledge}\", meaning [MASK].",
    "The statement \"{sentence}\" could indicate \"{knowledge}\", which implies [MASK].",
    "When considering \"{sentence}\", it can be linked to \"{knowledge}\", inferring [MASK].",
    "The sentence \"{sentence}\" appears related to \"{knowledge}\", implying [MASK].",
    "Understanding \"{sentence}\" in context with \"{knowledge}\" leads to [MASK].",
    "The expression \"{sentence}\" might reflect \"{knowledge}\", suggesting [MASK].",
    "Analyzing \"{sentence}\" with regard to \"{knowledge}\" infers that [MASK].",
    "The phrase \"{sentence}\" aligns with \"{knowledge}\", implying [MASK].",
    "Connecting \"{sentence}\" to \"{knowledge}\" might conclude that [MASK].",
]

def get_random_prompt():
    return random.choice(knowledge_prompt_list)
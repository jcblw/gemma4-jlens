"""Run several prompts through the lens with a single model load."""

from inspect_gemma import inspect_prompt, load

PROMPTS = [
    "Fact: The creator of the language that runs in every web browser now works at",
    "Fact: The capital of France is not",
    "Fact: The total number of legs on three spiders is",
    "Fact: The English word for the French word 'chien' is",
]

if __name__ == "__main__":
    model, tok, lens = load()
    for p in PROMPTS:
        print("\n" + "=" * 80)
        inspect_prompt(model, tok, lens, p)

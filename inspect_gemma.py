"""Inspect Gemma 4 E4B with Anthropic's Jacobian Lens.

Loads the HF weights (bf16 on MPS), applies the pre-fitted lens from
solarkyle/jspace-lenses, and prints what the residual stream is "poised
to verbalize" at each layer for a prompt.

Usage:
    uv run python inspect_gemma.py ["your prompt here"]
"""

import sys

import torch
import transformers
from huggingface_hub import hf_hub_download

import jlens

MODEL_ID = "google/gemma-4-e4b-it"
LENS_REPO = "solarkyle/jspace-lenses"
LENS_FILE = "gemma-4-e4b-it/lens.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    print(f"Loading {MODEL_ID} on {DEVICE} (bf16)...")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(DEVICE)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)

    lens_path = hf_hub_download(LENS_REPO, LENS_FILE)
    lens = jlens.JacobianLens.load(lens_path)
    return model, tok, lens


def inspect_prompt(model, tok, lens, prompt: str, position: int = -1, top_k: int = 8):
    print(f"\nPrompt: {prompt!r}  (reading position {position})")
    lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[position])

    print(f"\nModel's actual next-token top {top_k}:")
    for tid in model_logits[-1].topk(top_k).indices.tolist():
        print(f"  {tok.decode([tid])!r}")

    print(f"\nLens readout per layer (top {top_k} tokens the layer is poised to say):")
    for layer, logits in sorted(lens_logits.items()):
        vocab_logits = logits.reshape(-1, logits.size(-1))[-1]
        toks = [tok.decode([t]) for t in vocab_logits.topk(top_k).indices.tolist()]
        print(f"  L{layer:>2}: {toks}")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "Fact: The currency used in the country shaped like a boot is"
    )
    model, tok, lens = load()
    inspect_prompt(model, tok, lens, prompt)

# gemma4-jlens

Inspect local Gemma 4 models with [Anthropic's Jacobian Lens](https://github.com/anthropics/jacobian-lens) (`jlens`) — a tool that reads out what a transformer's internal activations are "poised to verbalize" by linearly transporting residual-stream vectors into the final-layer basis and decoding them with the model's own unembedding.

## Setup

Dependencies are managed with `uv` (already installed via `uv sync`):

- `jlens` from `git+https://github.com/anthropics/jacobian-lens`
- `torch` / `transformers` — the model runs in bf16 on MPS (Apple Silicon)
- Pre-fitted lens weights from [solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses)

Requires a HuggingFace login with access to the gated `google/gemma-4-e4b-it` repo.

## Usage

```bash
uv run python inspect_gemma.py "Fact: The capital of the country that gifted the Statue of Liberty is"
```

Prints the model's actual next-token predictions alongside the lens readout per layer — the ranked tokens each layer's residual stream is disposed to say.

## Notes

- **Model choice**: `gemma-4-e4b-it` (effective-4B). The 12B variant also has a pre-fitted lens (`gemma-4-12b-it/lens.pt`) but won't fit comfortably in 24 GB of unified memory in bf16.
- The Ollama installs (`gemma4:e4b`, `gemma4:12b-mlx`) are GGUF/MLX and can't be used by jlens — it needs HF `transformers` weights, downloaded separately to `~/.cache/huggingface`.
- To fit a custom lens instead of using the pre-fitted one: `jlens.fit(model, prompts=..., checkpoint_path=...)` — ~1000 sequences of 128 tokens recommended.
- Interactive explorer: https://neuronpedia.org/jlens

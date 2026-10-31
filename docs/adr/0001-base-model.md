# ADR 0001: Granite 4.1 3B as the base model

**Status:** accepted, 2026-10-30

## Context

The model has to train with LoRA on one Apple Silicon laptop (an M6 with 32 GB of unified memory) using `mlx_lm`. Its license must let us publish an adapter and a fused model. It should be small enough that "runs on a laptop" means fast, not barely. And it should be recent, so the comparison with frontier models isn't against an obsolete baseline.

These candidates were checked on Hugging Face (`mlx-community` conversions and the original repos) on 2026-10-30:

| Model | Params | License | Architecture in `mlx_lm` | Notes |
|:--|--:|:--|:--|:--|
| `ibm-granite/granite-4.1-3b` (Apr 2026) | 3B | Apache-2.0 | `GraniteForCausalLM`, plain dense | instruct model, tool calling advertised |
| `Qwen/Qwen3.5-2B` / `-4B` (Mar 2026) | 2B / 4B | Apache-2.0 | `Qwen3_5ForConditionalGeneration` (vision-language wrapper, hybrid attention) | thinking on by default |
| `Qwen/Qwen3-4B-Instruct-2507` (Aug 2025) | 4B | Apache-2.0 | dense | a year older |
| `google/gemma-4-E4B-it` (Apr 2026) | ~4B effective | Gemma terms | multimodal wrapper | custom license terms |
| `meta-llama/Llama-3.2-3B-Instruct` (Sep 2024) | 3B | Llama 3.2 Community | dense | old, custom license |
| `LiquidAI/LFM2.5-2.6B` (Aug 2026) | 2.6B | LFM Open License | hybrid conv/attention | custom license |

## Decision

Use `ibm-granite/granite-4.1-3b`, pinned to revision `c0650403e44e78ec0262dab1c90914c65b196c4e`. Train LoRA on the bf16 weights and quantize afterwards.

## Consequences

- Apache-2.0 means the adapter and the fused or quantized weights can be redistributed with attribution.
- The plain dense architecture is the best-trodden path through `mlx_lm` LoRA, fuse and convert. The Qwen 3.5 small models are wrapped as vision-language models with hybrid layers, which means more ways for training or quantization to break and nothing gained for a text task.
- Granite 4.1 was already post-trained for tool calling, so the zero-shot baseline is strong and gains from fine-tuning are harder to show. That makes for a stricter test of fine-tuning, not a reason to pick a weaker base.
- bf16 training needs about 10 GB peak at batch size 2 with gradient checkpointing (batch size 4 ran out of GPU memory on 32 GB). That leaves room for an eval process, though not a comfortable one (see the README's notes on running both at once).

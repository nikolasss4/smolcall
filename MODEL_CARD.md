---
license: apache-2.0
base_model: ibm-granite/granite-4.1-3b
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - lora
  - function-calling
  - tool-use
  - apple-silicon
datasets:
  - glaiveai/glaive-function-calling-v2
  - NousResearch/hermes-function-calling-v1
---

# smolcall: Granite 4.1 3B, LoRA fine-tuned for function calling (MLX)

A LoRA fine-tune of IBM's Granite 4.1 3B that turns a user request plus a list of tool schemas into a JSON array of tool calls, or `[]` when no tool fits. It was trained and evaluated on one Apple Silicon laptop with `mlx_lm`. Code, data pipeline, raw predictions and every number below: <https://github.com/nikolasss4/smolcall>.

## Model details

| | |
|:--|:--|
| Base model | [`ibm-granite/granite-4.1-3b`](https://huggingface.co/ibm-granite/granite-4.1-3b), revision `c0650403e44e78ec0262dab1c90914c65b196c4e`, Apache-2.0 |
| Method | LoRA, rank 16, every layer, `mlx_lm` scale 2.0, dropout 0, prompt tokens masked from the loss |
| Training | 1 epoch over 6,353 examples, batch 2 × gradient accumulation 2, Adam at 1e-4 (50 warmup steps, cosine to 1e-5), max length 2048, seed 0 |
| Variants | LoRA adapter; fused bf16; 8-bit and 4-bit (`mlx_lm.convert`, affine, group size 64; its default 4-bit recipe keeps each layer's attention output and MLP down projections at 8-bit) |
| Hardware | Apple M6, 32 GB unified memory. Main run: 106 min, peak 11.0 GB |
| License | Apache-2.0, inherited from the base model and both training datasets |

## Prompt format

The system message holds the instruction and one JSON Schema per tool. The user message is the request. The model answers with only a JSON array. `smolcall.prompt.system_prompt(tools)` builds the exact text, and `smolcall prompt --tools tools.json` prints it.

```text
[system] You are a function-calling assistant. ... Reply with only a JSON array of calls ...
         Available tools:
         {"name": "get_weather", "description": "...", "parameters": {...}}
[user]   What's the weather in Oslo, in fahrenheit?
[model]  [{"name": "get_weather", "arguments": {"city": "Oslo", "units": "fahrenheit"}}]
```

## Training data

| Source | License | Revision | Used for |
|:--|:--|:--|:--|
| [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | `e7f4b645` | train / val / test |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (`func-calling-singleturn`) | Apache-2.0 | `dae3e1d2` | train / val / test |
| [`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) v3 | Apache-2.0 | `61fc0608` | evaluation only |

The data was processed as follows:
- The first user turn of each conversation is taken, along with the gold call(s).
- Dropped: rows with unparseable arguments, calls to tools that weren't offered, empty-string argument values, and hermes "no call" replies (noisy).
- Rows are deduplicated by normalized request text, then capped at 40 examples per tool set.
- The test split holds out whole groups of co-offered tools, so every test tool is unseen in training. BFCL tools are also removed from train.
- The build fails on any leakage of ids, requests or held-out tools.

Both training datasets are synthetic: their authors generated them with other models. No data for this project was generated with Claude or any other frontier model.

## Evaluation

Exact match, AST-style:
- the right function names;
- the right argument names and values, with types checked (`"5"` is not `5`);
- calls matched order-insensitively;
- extra arguments allowed only when they equal the schema default.

BFCL's lists of accepted values are honoured. Intervals are 95% percentile bootstrap (2,000 resamples). Decoding is greedy.

| System | Held-out test (n=499, unseen tools) | Hand-checked (n=102) | BFCL v3 subset (n=300) |
|:--|--:|--:|--:|
| Granite 4.1 3B zero-shot, bf16 | 61.5% [57.3, 65.7] | 67.6% [58.8, 76.5] | **84.7%** [80.7, 88.7] |
| Granite 4.1 3B 3-shot, bf16 | 64.9% [60.7, 68.9] | 73.5% [65.7, 82.4] | 81.0% [76.3, 85.3] |
| Granite 4.1 3B zero-shot, 4-bit | 56.7% [52.3, 60.9] | 59.8% [50.0, 69.6] | 76.3% [71.3, 81.0] |
| **smolcall bf16** | **79.0%** [75.4, 82.6] | 89.2% [83.3, 95.1] | 80.7% [76.0, 85.0] |
| **smolcall 8-bit** | 78.6% [74.9, 82.4] | 89.2% [83.3, 95.1] | 80.0% [75.3, 84.3] |
| **smolcall 4-bit** | 77.4% [73.7, 81.0] | **90.2%** [84.3, 95.1] | 80.0% [75.3, 84.7] |

The frontier baselines (`claude-haiku-5-5`, `claude-sonnet-5-5`) are implemented in the repo but not run yet. See the README.

The 4-bit model on the M6, one request at a time, measured on the same 40 test requests:
- 1.09 s median end-to-end latency (7.6 s at p95);
- 66 tokens/s generation;
- 3.2 GB peak memory;
- 2.5 GB on disk.

## Intended use

Picking a tool and filling its arguments from one user request, inside an agent or app where the caller supplies the tool schemas and executes the calls. It is aimed at local or offline use where a 1 s, zero-API-cost call matters more than the last few points of accuracy.

## Limitations and failure modes

These were all observed in the committed predictions; see the README's error analysis.

- **No gain on BFCL.** The fine-tune does not beat the bf16 base on the BFCL subset (80.7% vs 84.7%, overlapping intervals). It calls a tool on 53% of BFCL's "irrelevance" questions, against 33% for the base. In-distribution, refusing went from bad to good: the false-call rate fell from 71% to 3%. That did not transfer to BFCL's style of near-miss tools.
- **Refuses argument-less tools.** It answers `[]` to "I need a new invoice number" when `generate_invoice_number()` takes no arguments. glaive's training labels often do the same.
- **Runs ahead in dependent chains.** When step 2 needs an id that step 1 returns, it sometimes calls step 2 anyway with a guessed value.
- **Paraphrases enum-like strings.** For example `household_size` for "household size", or `Bengal Tiger` for "Bengal Tigers".
- **Single turn only.** It has not been trained or evaluated on multi-turn conversations or on reading tool results.
- **English only** in training and evaluation.
- The automatic test labels are themselves noisy. Of 150 reviewed, 48 were unusable and 5 were wrong, so read the held-out test numbers together with the hand-checked set.

## Compute and carbon

All training and evaluation ran on one laptop. The main run took 106 minutes. Power wasn't measured (that needs root). At an assumed 30 W whole-machine draw, the main run used about 0.05 kWh. The ablation and evaluation runs add a few times that.

## Citation and attribution

Base model © IBM, Apache-2.0. Training data © Glaive AI and Nous Research, Apache-2.0. Evaluation data © the Gorilla / BFCL authors, Apache-2.0.

.PHONY: install data train fuse quantize eval eval-local eval-frontier bench ablations charts rescore test lint serve demo

BASE := ibm-granite/granite-4.1-3b
LOCAL_SYSTEMS := base base-fewshot ft ft-8bit ft-4bit base-4bit
FRONTIER_SYSTEMS := claude-haiku-5-5 claude-sonnet-5-5

install:  ## Python deps, including MLX (Apple Silicon only)
	uv sync --all-extras

data:  ## Download pinned sources, verify hashes, build leakage-checked splits
	uv run smolcall data download
	uv run smolcall data build

train:  ## The headline LoRA run (about 90 min on an M6 / 32 GB)
	uv run smolcall train configs/main.yaml

fuse:  ## Merge the adapter into the bf16 base
	uv run mlx_lm.fuse --model $(BASE) --adapter-path runs/main/adapters --save-path models/smolcall-bf16

quantize: fuse  ## 8-bit and 4-bit versions of the fine-tune, and a 4-bit base for comparison
	uv run mlx_lm.convert --hf-path models/smolcall-bf16 --mlx-path models/smolcall-8bit -q --q-bits 8
	uv run mlx_lm.convert --hf-path models/smolcall-bf16 --mlx-path models/smolcall-4bit -q --q-bits 4
	uv run mlx_lm.convert --hf-path $(BASE) --mlx-path models/smolcall-base-4bit -q --q-bits 4

eval-local:
	uv run smolcall eval $(foreach s,$(LOCAL_SYSTEMS),-s $(s))

eval-frontier:  ## Needs ANTHROPIC_API_KEY; about $3 for both models on all splits
	uv run smolcall eval $(foreach s,$(FRONTIER_SYSTEMS),-s $(s))

eval: eval-local eval-frontier

bench:
	uv run smolcall bench $(foreach s,$(LOCAL_SYSTEMS),-s $(s))

ablations:  ## Learning curve and LoRA rank runs, each evaluated on the test split
	for c in configs/ablations/*.yaml; do uv run smolcall train $$c; done
	for c in configs/ablations/*.yaml; do uv run smolcall eval -s lora-$$(basename $$c .yaml) --split test --split handchecked; done

charts:
	uv run python scripts/charts.py

rescore:  ## Re-score every committed prediction file (no GPU or API key needed)
	uv run python scripts/rescore.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

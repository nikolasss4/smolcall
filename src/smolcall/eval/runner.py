"""Run a system on a split, save every raw prediction, and write the metrics.

Predictions go to ``results/predictions/<system>/<split>.jsonl`` (committed, so every
number in the README can be re-scored without a GPU or an API key) and metrics to
``results/metrics/<system>__<split>.json``. Runs resume: examples that already have a
prediction are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smolcall.data.build import DATA, PROCESSED, ROOT, read_jsonl
from smolcall.eval.metrics import aggregate
from smolcall.eval.scorer import Score, score_example
from smolcall.eval.systems import MLXSystem, Prediction, ReplaySystem, System
from smolcall.types import Example

RESULTS = ROOT / "results"
BASE_MODEL = "ibm-granite/granite-4.1-3b"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
SPLITS = ("test", "handchecked", "bfcl")


def load_split(split: str) -> list[Example]:
    path = DATA / "handchecked.jsonl" if split == "handchecked" else PROCESSED / f"{split}.jsonl"
    return read_jsonl(path)


def prediction_path(system: str, split: str) -> Path:
    return RESULTS / "predictions" / system / f"{split}.jsonl"


def get_system(name: str, split: str) -> System:
    from smolcall.prompt import pick_shots

    if name == "base":
        return MLXSystem(name, BASE_MODEL)
    if name == "base-fewshot":
        return MLXSystem(name, BASE_MODEL, shots=pick_shots(read_jsonl(PROCESSED / "train.jsonl")))
    if name in ("ft", "ft-8bit", "ft-4bit", "base-4bit"):
        suffix = {"ft": "bf16", "ft-8bit": "8bit", "ft-4bit": "4bit", "base-4bit": "base-4bit"}
        return MLXSystem(name, str(MODELS / f"smolcall-{suffix[name]}"))
    if name.startswith("lora-"):
        # An ablation run, evaluated as base + adapter without fusing.
        run = name.removeprefix("lora-")
        return MLXSystem(name, BASE_MODEL, adapter=str(RUNS / run / "adapters"))
    if name.startswith("claude-"):
        from smolcall.eval.frontier import AnthropicSystem

        return AnthropicSystem(name, name)
    if name.startswith("replay:"):
        recorded = name.removeprefix("replay:")
        return ReplaySystem(recorded, prediction_path(recorded, split))
    raise ValueError(f"unknown system {name!r}")


def _read_predictions(path: Path) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    preds = (Prediction.from_dict(json.loads(line)) for line in path.read_text().splitlines())
    return {p.id: p for p in preds}


def summarize(
    system: str, split: str, examples: list[Example], preds: dict[str, Prediction]
) -> dict[str, Any]:
    scores: list[Score] = [score_example(preds[e.id].calls, e) for e in examples]
    out = {"system": system, "split": split, **aggregate(scores, examples)}
    usage = [preds[e.id].usage for e in examples]
    if all("cost_usd" in u for u in usage):
        total = sum(u["cost_usd"] for u in usage)
        out["cost_usd_total"] = round(total, 6)
        out["cost_usd_per_1k_calls"] = round(1000 * total / len(usage), 4)
        lat = sorted(u["latency_s"] for u in usage)
        out["latency_s_p50"] = lat[len(lat) // 2]
        out["latency_s_p95"] = lat[int(len(lat) * 0.95)]
        out["output_tokens_mean"] = sum(u["output_tokens"] for u in usage) / len(usage)
    return out


def run(system_name: str, split: str, limit: int | None = None) -> dict[str, Any]:
    examples = load_split(split)[:limit]
    replay = system_name.startswith("replay:")
    system = get_system(system_name, split)
    name = system.name
    if replay:
        preds = {p.id: p for p in system.predict(examples)}
    else:
        path = prediction_path(name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        preds = _read_predictions(path)
        todo = [e for e in examples if e.id not in preds]
        with path.open("a") as f:
            for i, p in enumerate(system.predict(todo), 1):
                preds[p.id] = p
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"{name} {split}: {i}/{len(todo)}", flush=True)
    metrics = summarize(name, split, examples, preds)
    if not replay and limit is None:
        out = RESULTS / "metrics" / f"{name}__{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics

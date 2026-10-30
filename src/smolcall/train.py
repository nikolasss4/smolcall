"""LoRA training with mlx_lm, driven by a small YAML config.

``configs/<name>.yaml`` holds only what differs from mlx_lm's defaults plus two smolcall keys:

- ``train_size``: use the first N examples of a fixed shuffle of the train split, so the
  learning-curve subsets are nested (the 1k set is inside the 3k set);
- ``epochs``: sets ``iters`` to ``ceil(epochs * train_size / batch_size)``.

Everything lands in ``runs/<name>/``: the resolved config, the adapter, and
``loss.jsonl`` with every train and validation loss report.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import yaml

from smolcall.data.build import PROCESSED, ROOT, read_jsonl
from smolcall.prompt import training_record

RUNS = ROOT / "runs"


def load_config(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    cfg.setdefault("name", path.stem)
    return cfg


def prepare_data(train_size: int | None, seed: int, out_dir: Path) -> int:
    """Write mlx_lm chat-format train/valid JSONL; returns the number of train examples."""
    train = read_jsonl(PROCESSED / "train.jsonl")
    random.Random(seed).shuffle(train)
    if train_size:
        train = train[:train_size]
    val = read_jsonl(PROCESSED / "val.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, exs in (("train", train), ("valid", val)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for e in exs:
                f.write(json.dumps(training_record(e), ensure_ascii=False) + "\n")
    return len(train)


def resolve(cfg: dict[str, Any], n_train: int, run_dir: Path, data_dir: Path) -> dict[str, Any]:
    from mlx_lm.lora import CONFIG_DEFAULTS

    args = {**CONFIG_DEFAULTS, **{k: v for k, v in cfg.items() if k in CONFIG_DEFAULTS}}
    args.update(train=True, data=str(data_dir), adapter_path=str(run_dir / "adapters"))
    if "epochs" in cfg:
        args["iters"] = math.ceil(cfg["epochs"] * n_train / args["batch_size"])
    return args


class JsonlLogger:
    """mlx_lm TrainingCallback that appends every loss report to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.start = time.time()
        path.write_text("")

    def _write(self, kind: str, info: dict[str, Any]) -> None:
        row = {"kind": kind, "elapsed_s": round(time.time() - self.start, 1)}
        row.update({k: float(v) if isinstance(v, int | float) else v for k, v in info.items()})
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        self._write("train", train_info)

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        self._write("val", val_info)


def train(config_path: Path) -> Path:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.lora import load, load_dataset, train_model

    cfg = load_config(config_path)
    run_dir = RUNS / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    n_train = prepare_data(cfg.get("train_size"), cfg.get("seed", 0), data_dir)
    args = resolve(cfg, n_train, run_dir, data_dir)
    (run_dir / "config.json").write_text(
        json.dumps({"smolcall": cfg, "mlx_lm": args, "n_train": n_train}, indent=2) + "\n"
    )

    mx.random.seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    if "cosine_warmup" in cfg:
        # Linear warmup then cosine decay to 10% of the peak over the whole run.
        lr = args["learning_rate"]
        args["lr_schedule"] = {
            "name": "cosine_decay",
            "arguments": [lr, args["iters"] - cfg["cosine_warmup"], lr * 0.1],
            "warmup": cfg["cosine_warmup"],
        }
    ns = _Namespace(args)
    start = time.time()
    # mlx_lm.lora.run() replaces any callback it is given with its own reporters, so this
    # mirrors its train path to keep our loss log.
    model, tokenizer = load(args["model"])
    train_set, valid_set, _ = load_dataset(ns, tokenizer)
    train_model(ns, model, train_set, valid_set, JsonlLogger(run_dir / "loss.jsonl"))
    summary = {
        "n_train": n_train,
        "iters": args["iters"],
        "wall_time_s": round(time.time() - start, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir


class _Namespace:
    """argparse.Namespace stand-in that mlx_lm.lora.run accepts."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.__dict__.update(d)

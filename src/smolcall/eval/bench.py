"""Single-request latency, throughput, memory and disk size of a local model.

Accuracy runs batch requests; this measures what one user sees: one request at a time,
greedy, on the same fixed slice of the test split for every model. Reported per model:
p50/p95 end-to-end latency, prompt and generation tokens/sec, peak memory and size on disk.

Local "cost" is electricity only. We can't read package power without root, so it is an
estimate from a stated assumption (``ASSUMED_WATTS``), not a measurement.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from smolcall.eval.runner import RESULTS, get_system, load_split
from smolcall.eval.systems import MLXSystem

ASSUMED_WATTS = 30.0  # whole-machine draw while generating; an assumption, see README
USD_PER_KWH = 0.30


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def model_dir(model: str) -> Path:
    p = Path(model)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"]))


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def bench(system_name: str, n: int = 40, warmup: int = 3) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    system = get_system(system_name, "test")
    if not isinstance(system, MLXSystem):
        raise ValueError("bench only measures local MLX models")
    model, tokenizer = system.load()
    examples = load_split("test")[: n + warmup]
    rows = []
    mx.reset_peak_memory()
    for i, ex in enumerate(examples):
        prompt = system.prompt_tokens(ex)
        start = time.perf_counter()
        last = None
        for response in stream_generate(model, tokenizer, prompt, max_tokens=system.max_tokens):
            last = response
        latency = time.perf_counter() - start
        if i < warmup or last is None:
            continue
        rows.append(
            {
                "latency_s": latency,
                "prompt_tokens": last.prompt_tokens,
                "prompt_tps": last.prompt_tps,
                "generation_tokens": last.generation_tokens,
                "generation_tps": last.generation_tps,
            }
        )
    lat = [r["latency_s"] for r in rows]
    disk = dir_size_gb(model_dir(system.model_path))
    if system.adapter:
        disk += dir_size_gb(Path(system.adapter))
    kwh_per_call = ASSUMED_WATTS * statistics.mean(lat) / 3600 / 1000
    out = {
        "system": system_name,
        "n": len(rows),
        "latency_s_p50": _pct(lat, 0.5),
        "latency_s_p95": _pct(lat, 0.95),
        "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in rows),
        "prompt_tps_mean": statistics.mean(r["prompt_tps"] for r in rows),
        "generation_tps_mean": statistics.mean(r["generation_tps"] for r in rows),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "disk_gb": round(disk, 2),
        "electricity_usd_per_1k_calls_est": round(1000 * kwh_per_call * USD_PER_KWH, 4),
        "assumptions": {"watts": ASSUMED_WATTS, "usd_per_kwh": USD_PER_KWH},
    }
    path = RESULTS / "bench" / f"{system_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out

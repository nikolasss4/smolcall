"""Aggregate per-example scores into the numbers reported in the README."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from smolcall.eval.scorer import ERRORS, Score
from smolcall.types import Example


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int(n_boot * alpha / 2)]
    hi = means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]
    return (lo, hi)


def _rate(flags: list[bool], with_ci: bool = True) -> dict[str, Any]:
    vals = [float(f) for f in flags]
    out: dict[str, Any] = {"n": len(vals), "value": sum(vals) / len(vals) if vals else None}
    if with_ci and vals:
        out["ci95"] = list(bootstrap_ci(vals))
    return out


def aggregate(scores: Sequence[Score], examples: Sequence[Example]) -> dict[str, Any]:
    if len(scores) != len(examples):
        raise ValueError("scores and examples must line up")
    pairs = list(zip(scores, examples, strict=True))
    none = [s for s, e in pairs if e.category == "none"]
    calls = [s for s, e in pairs if e.category != "none"]
    return {
        "n": len(pairs),
        "accuracy": _rate([s.correct for s in scores]),
        "json_valid": _rate([s.json_valid for s in scores]),
        "name_match": _rate([s.name_match for s in calls]),
        "single_accuracy": _rate([s.correct for s, e in pairs if e.category == "single"]),
        "multi_accuracy": _rate([s.correct for s, e in pairs if e.category == "multi"]),
        "no_call_accuracy": _rate([s.correct for s in none]),
        # Of the requests that needed no tool, how often the system called one anyway.
        "false_call_rate": _rate([s.error == "over_call" for s in none]),
        # Of the requests that needed a tool, how often the system called nothing.
        "missed_call_rate": _rate([s.error == "missed_call" for s in calls]),
        "errors": {k: v for k, v in Counter(s.error for s in scores if s.error).items()},
    }


def error_table(scores: Sequence[Score]) -> dict[str, int]:
    counts = Counter(s.error for s in scores if s.error)
    return {e: counts[e] for e in ERRORS if counts[e]}

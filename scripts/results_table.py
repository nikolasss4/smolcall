"""Print the README's results tables from results/metrics and results/bench.

    uv run python scripts/results_table.py

Every number in the README's results section comes from this script's output.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
BENCH = ROOT / "results" / "bench"

SYSTEMS = [
    ("base", "Granite 4.1 3B, zero-shot (bf16)"),
    ("base-fewshot", "Granite 4.1 3B, 3-shot (bf16)"),
    ("base-4bit", "Granite 4.1 3B, zero-shot (4-bit)"),
    ("ft", "**smolcall** LoRA fine-tune (bf16)"),
    ("ft-8bit", "**smolcall** fine-tune (8-bit)"),
    ("ft-4bit", "**smolcall** fine-tune (4-bit)"),
    ("claude-haiku-5-5", "claude-haiku-5-5 (native tool use)"),
    ("claude-sonnet-5-5", "claude-sonnet-5-5 (native tool use)"),
]
SPLITS = [
    ("test", "Held-out test (499)"),
    ("handchecked", "Hand-checked (102)"),
    ("bfcl", "BFCL subset (300)"),
]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def acc(m: dict | None, key: str = "accuracy") -> str:
    if not m or m[key]["value"] is None:
        return "_pending_"
    v = m[key]["value"]
    lo, hi = m[key]["ci95"]
    return f"{v:.1%} <sub>[{lo:.1%}, {hi:.1%}]</sub>"


def pct(m: dict | None, key: str) -> str:
    return "_pending_" if not m or m[key]["value"] is None else f"{m[key]['value']:.1%}"


def main() -> None:
    print("| System | " + " | ".join(label for _, label in SPLITS) + " |")
    print("|:--|" + "--:|" * len(SPLITS))
    for sys_name, label in SYSTEMS:
        cells = [acc(load(METRICS / f"{sys_name}__{sp}.json")) for sp, _ in SPLITS]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print(
        "\n| System | p50 latency | p95 latency | gen tok/s | peak memory | size on disk | cost / 1k calls |"
    )
    print("|:--|--:|--:|--:|--:|--:|--:|")
    for sys_name, label in SYSTEMS:
        b = load(BENCH / f"{sys_name}.json")
        if b:
            print(
                f"| {label} | {b['latency_s_p50']:.2f} s | {b['latency_s_p95']:.2f} s | "
                f"{b['generation_tps_mean']:.0f} | {b['peak_memory_gb']:.1f} GB | {b['disk_gb']:.1f} GB | "
                f"~${b['electricity_usd_per_1k_calls_est']:.3f} (electricity, est.) |"
            )
            continue
        m = load(METRICS / f"{sys_name}__test.json")
        if sys_name.startswith("claude"):
            if m and "cost_usd_per_1k_calls" in m:
                print(
                    f"| {label} | {m['latency_s_p50']:.2f} s | {m['latency_s_p95']:.2f} s | - | - | - | "
                    f"${m['cost_usd_per_1k_calls']:.2f} (API) |"
                )
            else:
                print(f"| {label} | _pending_ | _pending_ | - | - | - | _pending_ |")

    print("\nBreakdown on the held-out test split:\n")
    print(
        "| System | JSON valid | right tool names | multi-call | no-tool correct | false-call rate |"
    )
    print("|:--|--:|--:|--:|--:|--:|")
    for sys_name, label in SYSTEMS:
        m = load(METRICS / f"{sys_name}__test.json")
        if not m:
            continue
        print(
            f"| {label} | {pct(m, 'json_valid')} | {pct(m, 'name_match')} | {pct(m, 'multi_accuracy')} | "
            f"{pct(m, 'no_call_accuracy')} | {pct(m, 'false_call_rate')} |"
        )


if __name__ == "__main__":
    main()

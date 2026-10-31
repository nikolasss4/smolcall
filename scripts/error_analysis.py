"""Group each system's failures by kind and pull real examples for the write-up.

    uv run python scripts/error_analysis.py ft base
    uv run python scripts/error_analysis.py ft --examples 3 --split handchecked

Prints a per-category count table for every system (all three eval splits pooled, or one
split) and, for the first system, a few failing examples per category with the request,
the gold calls and what the model said. Reads only committed predictions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from smolcall.eval.runner import SPLITS, load_split, prediction_path, run
from smolcall.eval.scorer import ERRORS, score_example
from smolcall.eval.systems import ReplaySystem


def failures(system: str, splits: list[str]) -> list[tuple[str, object, object, str]]:
    out = []
    for split in splits:
        if not prediction_path(system, split).exists():
            continue
        examples = load_split(split)
        preds = {
            p.id: p for p in ReplaySystem(system, prediction_path(system, split)).predict(examples)
        }
        for ex in examples:
            s = score_example(preds[ex.id].calls, ex)
            if not s.correct:
                out.append((split, ex, preds[ex.id], s.error or "wrong_value"))
    return out


def short(obj: object, n: int = 400) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()
    splits = [args.split] if args.split else list(SPLITS)

    counts = {s: Counter(e for *_, e in failures(s, splits)) for s in args.systems}
    totals = {s: sum(run(f"replay:{s}", sp)["n"] for sp in splits) for s in args.systems}
    print("| error | " + " | ".join(args.systems) + " |")
    print("|:--|" + "--:|" * len(args.systems))
    for err in ERRORS:
        if any(counts[s][err] for s in args.systems):
            print(f"| {err} | " + " | ".join(str(counts[s][err]) for s in args.systems) + " |")
    print(
        "| **failures / examples** | "
        + " | ".join(f"{sum(counts[s].values())} / {totals[s]}" for s in args.systems)
        + " |"
    )

    rng = random.Random(0)
    by_kind: dict[str, list] = {}
    for f in failures(args.systems[0], splits):
        by_kind.setdefault(f[3], []).append(f)
    for err in ERRORS:
        if err not in by_kind:
            continue
        print(f"\n### {err} ({len(by_kind[err])})\n")
        for split, ex, pred, _ in rng.sample(by_kind[err], min(args.examples, len(by_kind[err]))):
            print(f"- `{ex.id}` ({split}) request: {short(ex.query, 300)!r}")
            gold = [{"name": c.name, "arguments": c.arguments} for c in ex.calls]
            print(f"  - gold: `{short(gold)}`")
            print(f"  - model: `{short(pred.text.strip())}`")


if __name__ == "__main__":
    main()

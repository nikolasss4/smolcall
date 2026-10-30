"""Re-score every committed prediction file and check it matches the committed metrics.

This is what CI runs: the README numbers come from results/metrics/*.json, and those must
be reproducible from the raw model outputs in results/predictions/ with the current
scorer. No GPU or API key is needed.
"""

import json
import sys

from smolcall.eval.runner import RESULTS, run

KEYS = ("accuracy", "json_valid", "name_match", "multi_accuracy", "false_call_rate")


def main() -> int:
    failures = 0
    for path in sorted((RESULTS / "metrics").glob("*__*.json")):
        system, split = path.stem.split("__")
        committed = json.loads(path.read_text())
        fresh = run(f"replay:{system}", split)
        diffs = [k for k in KEYS if committed[k]["value"] != fresh[k]["value"]]
        status = "ok  " if not diffs else "DIFF"
        print(f"{status} {system:<24} {split:<12} acc={fresh['accuracy']['value']:.3f}")
        if diffs:
            failures += 1
            for k in diffs:
                print(
                    f"     {k}: committed {committed[k]['value']} != rescored {fresh[k]['value']}"
                )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

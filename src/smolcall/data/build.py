"""Download pinned raw data and build the processed splits.

``data/manifest.json`` pins every source to a Hugging Face revision and a SHA-256 per file,
so a rebuild either reproduces the same splits or fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from smolcall.data.normalize import parse_bfcl, parse_glaive, parse_hermes
from smolcall.data.split import cap_per_toolset, check_leakage, split
from smolcall.types import Example

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.json"

# Train mix and split sizes. Changing any of these changes the splits, so bump the salt
# in split() too if you want old and new results to be clearly distinct.
TRAIN_CAP_PER_TOOLSET = 40
TEST_CAP_PER_TOOLSET = 2
TEST_SIZE = 500
VAL_SIZE = 300
BFCL_PER_CATEGORY = 60
BFCL_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
SEED = 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def download() -> None:
    from huggingface_hub import hf_hub_download

    for source in load_manifest()["sources"]:
        for file, expected in source["files"].items():
            local = Path(
                hf_hub_download(
                    source["repo"],
                    file,
                    repo_type="dataset",
                    revision=source["revision"],
                    local_dir=RAW / source["name"],
                )
            )
            actual = sha256(local)
            if actual != expected:
                raise SystemExit(f"{source['name']}/{file}: sha256 {actual} != pinned {expected}")
            print(f"ok  {source['name']}/{file}")


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_training_sources() -> list[Example]:
    glaive = json.loads((RAW / "glaive/glaive-function-calling-v2.json").read_text())
    hermes = json.loads((RAW / "hermes/func-calling-singleturn.json").read_text())
    out = [e for i, row in enumerate(glaive) if (e := parse_glaive(row, i))]
    # hermes "no call" replies are mostly the model asking for files or narrating a plan it
    # could have executed, so they are too noisy to teach refusals. Keep its calls only.
    out += [e for row in hermes if (e := parse_hermes(row)) and e.calls]
    return out


def load_bfcl() -> list[Example]:
    rng = random.Random(SEED)
    out = []
    for cat in BFCL_CATEGORIES:
        questions = _read_json_lines(RAW / f"bfcl/BFCL_v3_{cat}.json")
        answers_path = RAW / f"bfcl/possible_answer/BFCL_v3_{cat}.json"
        answers = (
            {a["id"]: a for a in _read_json_lines(answers_path)} if answers_path.exists() else {}
        )
        picked = rng.sample(questions, min(BFCL_PER_CATEGORY, len(questions)))
        out += [
            parse_bfcl(q, answers.get(q["id"]), cat) for q in sorted(picked, key=lambda q: q["id"])
        ]
    return out


def _stratified_sample(examples: list[Example], n: int, rng: random.Random) -> list[Example]:
    """Sample n examples keeping the none/single/multi proportions of the pool."""
    by_cat: dict[str, list[Example]] = {}
    for e in examples:
        by_cat.setdefault(e.category, []).append(e)
    total = len(examples)
    out = []
    for cat in sorted(by_cat):
        k = round(n * len(by_cat[cat]) / total)
        out += rng.sample(by_cat[cat], min(k, len(by_cat[cat])))
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in examples:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    return [Example.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def build() -> dict[str, Any]:
    rng = random.Random(SEED)
    examples = load_training_sources()
    splits = split(examples)
    bfcl = load_bfcl()
    # BFCL is an outside benchmark: also keep its tools out of train and val.
    bfcl_tools = {n for e in bfcl for n in e.tool_names}
    for name in ("train", "val"):
        splits[name] = [e for e in splits[name] if not bfcl_tools.intersection(e.tool_names)]
    splits["train"] = cap_per_toolset(splits["train"], TRAIN_CAP_PER_TOOLSET, SEED)
    splits["val"] = cap_per_toolset(splits["val"], TRAIN_CAP_PER_TOOLSET, SEED)[:VAL_SIZE]
    splits["test"] = _stratified_sample(
        cap_per_toolset(splits["test"], TEST_CAP_PER_TOOLSET, SEED), TEST_SIZE, rng
    )
    splits["bfcl"] = bfcl

    problems = check_leakage(splits, heldout=("test", "bfcl"))
    for name, exs in splits.items():
        for e in exs:
            offered = set(e.tool_names)
            problems += [
                f"{e.id} in {name}: gold calls {c.name!r}, which is not offered"
                for c in e.calls
                if c.name not in offered
            ]
    if problems:
        raise SystemExit("data problems:\n" + "\n".join(problems[:20]))

    stats: dict[str, Any] = {"parsed": len(examples), "splits": {}}
    for name, exs in splits.items():
        path = PROCESSED / f"{name}.jsonl"
        write_jsonl(path, exs)
        stats["splits"][name] = {
            "n": len(exs),
            "categories": dict(Counter(e.category for e in exs)),
            "sources": dict(Counter(e.source for e in exs)),
            "distinct_tools": len({n for e in exs for n in e.tool_names}),
            "sha256": sha256(path),
        }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats

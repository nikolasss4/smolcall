"""Deduplication, leakage-safe splits and the leakage checker.

The test split holds out whole tools. Tools that are offered together in any example are
linked, and each connected group of tools goes entirely to test or entirely to train/val,
decided by a salted hash so the split is stable and needs no state. Holding out single tool
names instead would cascade: one test example that also offers ``calculate_bmi`` would pull
every ``calculate_bmi`` example out of train. Groups bigger than ``max_test_group`` examples
(in glaive, one group links most of the popular tools) always stay in train.

On top of that, no normalized query may appear in more than one split.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from smolcall.types import Example

_WS = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    return _WS.sub(" ", query.casefold()).strip(" .!?")


def _unit(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def dedup(examples: Iterable[Example]) -> list[Example]:
    """Keep the first example for each normalized query."""
    seen: set[str] = set()
    out = []
    for ex in examples:
        key = normalize_query(ex.query)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def tool_groups(examples: list[Example]) -> dict[str, str]:
    """Map each tool name to a representative of its co-occurrence group (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ex in examples:
        names = ex.tool_names
        for n in names:
            find(n)
        for n in names[1:]:
            a, b = find(names[0]), find(n)
            if a != b:
                # Keep the alphabetically smallest name as root so roots are stable.
                parent[max(a, b)] = min(a, b)
    return {n: find(n) for n in parent}


def split(
    examples: list[Example],
    test_group_frac: float = 0.5,
    max_test_group: int = 50,
    val_frac: float = 0.03,
    salt: str = "smolcall-v1",
) -> dict[str, list[Example]]:
    examples = [e for e in dedup(examples) if e.tools]
    group_of = tool_groups(examples)
    sizes = Counter(group_of[e.tool_names[0]] for e in examples)
    test_groups = {
        g for g, n in sizes.items() if n <= max_test_group and _unit(g, salt) < test_group_frac
    }
    test: list[Example] = []
    rest: list[Example] = []
    for e in examples:
        (test if group_of[e.tool_names[0]] in test_groups else rest).append(e)
    val = [e for e in rest if _unit(e.id, salt + ":val") < val_frac]
    val_ids = {e.id for e in val}
    train = [e for e in rest if e.id not in val_ids]
    return {"train": train, "val": val, "test": test}


def check_leakage(
    splits: dict[str, list[Example]], heldout: Iterable[str] = ("test",)
) -> list[str]:
    """Return human-readable problems; an empty list means the splits are clean.

    ``heldout`` names the splits whose tools must never be offered in any other split.
    """
    problems = []
    ids: dict[str, str] = {}
    queries: dict[str, str] = {}
    for name, exs in splits.items():
        for ex in exs:
            if ex.id in ids:
                problems.append(f"id {ex.id} is in both {ids[ex.id]} and {name}")
            ids[ex.id] = name
            q = normalize_query(ex.query)
            if q in queries and queries[q] != name:
                problems.append(f"query {q[:60]!r} is in both {queries[q]} and {name}")
            queries.setdefault(q, name)
    for held in heldout:
        held_names = {n for e in splits.get(held, []) for n in e.tool_names}
        for name, exs in splits.items():
            if name in heldout:
                continue
            leaked = held_names.intersection(n for e in exs for n in e.tool_names)
            for tool in sorted(leaked):
                problems.append(f"tool {tool!r} from {held} is also offered in {name}")
    return problems


def cap_per_toolset(examples: list[Example], k: int, seed: int = 0) -> list[Example]:
    """Keep at most ``k`` examples per distinct set of offered tools, chosen at random.

    glaive repeats a few hundred tools thousands of times; capping trades volume for variety.
    """
    groups: dict[tuple[str, ...], list[Example]] = defaultdict(list)
    for ex in examples:
        groups[tuple(sorted(ex.tool_names))].append(ex)
    rng = random.Random(seed)
    out = []
    for key in sorted(groups):
        group = groups[key]
        out.extend(group if len(group) <= k else rng.sample(group, k))
    rng.shuffle(out)
    return out

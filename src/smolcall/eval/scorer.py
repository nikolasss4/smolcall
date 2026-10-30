"""Parse model output into tool calls and score it against gold, AST-style.

Every system (local models and frontier APIs) goes through ``score_example``, so the
numbers in the results table are comparable. The rules:

- Calls match order-insensitively (a maximum bipartite matching between predicted and gold
  calls); arguments match by name, order-insensitively.
- Values compare by type and value: ``5 == 5.0``, but ``"5" != 5`` and ``True != 1``.
  Strings compare after case-folding and collapsing whitespace. Lists are ordered.
- An argument the gold omits is allowed only if it equals the schema default.
- BFCL gold lists accepted values per argument; ``""`` among them means "may be omitted".
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from smolcall.types import Call, Example

# Most specific first: when a multi-call answer has several problems we report one.
ARG_ERRORS = ("hallucinated_arg", "missing_arg", "extra_arg", "wrong_type", "wrong_value")
ERRORS = (
    "invalid_json",
    "over_call",
    "missed_call",
    "wrong_tool",
    "missed_multi_call",
    "extra_call",
    *ARG_ERRORS,
)


@dataclass
class Score:
    correct: bool
    json_valid: bool
    name_match: bool
    error: str | None


# --- parsing ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _as_call(obj: Any) -> Call | None:
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):  # OpenAI-style wrapper
        obj = obj["function"]
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        return None
    return Call(name, args)


def _as_calls(obj: Any) -> list[Call] | None:
    if isinstance(obj, dict) and "tool_calls" in obj:
        obj = obj["tool_calls"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return None
    calls = [_as_call(o) for o in obj]
    return None if any(c is None for c in calls) else [c for c in calls if c]


def parse_output(text: str) -> list[Call] | None:
    """Return the calls in a model's text output, ``[]`` for "no tool", ``None`` if the
    output isn't a recognizable list of calls.

    A reply that contains no JSON at all ("Which city do you mean?") counts as calling
    nothing, the same as a frontier model answering in text instead of using a tool.
    Malformed JSON is ``None``.
    """
    text = _THINK.sub("", text).strip()
    if "{" not in text and "[" not in text:
        return []
    candidates = [text]
    candidates += _FENCE.findall(text)
    for open_, close in (("[", "]"), ("{", "}")):
        start, end = text.find(open_), text.rfind(close)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        calls = _as_calls(obj)
        if calls is not None:
            return calls
    return _parse_object_sequence(text)


def _parse_object_sequence(text: str) -> list[Call] | None:
    """Calls written one after another without one enclosing array: one JSON object per
    line, or one array per call (``[{...}], [{...}]``). Small models do this often; we
    accept it rather than count a readable answer as malformed."""
    decoder = json.JSONDecoder()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    calls: list[Call] = []
    i = min(starts)
    while i < len(text):
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return None
        part = _as_calls(obj)
        if part is None:
            return None
        calls.extend(part)
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
    return calls


# --- values ----------------------------------------------------------------------------


def _kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int | float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list | tuple):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _norm_str(s: str) -> str:
    return " ".join(s.casefold().split())


def values_equal(pred: Any, gold: Any) -> str | None:
    """``None`` if equal, else ``"wrong_type"`` or ``"wrong_value"``."""
    kp, kg = _kind(pred), _kind(gold)
    if kp != kg:
        return "wrong_type"
    if kp == "number":
        return None if math.isclose(pred, gold, rel_tol=1e-9, abs_tol=1e-12) else "wrong_value"
    if kp == "string":
        return None if _norm_str(pred) == _norm_str(gold) else "wrong_value"
    if kp == "list":
        if len(pred) != len(gold):
            return "wrong_value"
        errs = [values_equal(p, g) for p, g in zip(pred, gold, strict=True)]
        return next((e for e in errs if e), None)
    if kp == "dict":
        if set(pred) != set(gold):
            return "wrong_value"
        errs = [values_equal(pred[k], gold[k]) for k in gold]
        return next((e for e in errs if e), None)
    return None if pred == gold else "wrong_value"


def _matches_alternative(pred: Any, alt: Any) -> str | None:
    """Like values_equal, but a BFCL dict alternative maps each key to its own list of
    accepted values."""
    if isinstance(alt, dict) and isinstance(pred, dict):
        for k, v in alt.items():
            options = v if isinstance(v, list) else [v]
            if k not in pred:
                if "" in options:
                    continue
                return "wrong_value"
            if values_equal(pred[k], v) is None:
                continue
            if not any(values_equal(pred[k], o) is None for o in options if o != ""):
                return "wrong_value"
        return None if set(pred) <= set(alt) else "wrong_value"
    return values_equal(pred, alt)


def _value_error(pred: Any, accepted: list[Any], bfcl: bool) -> str | None:
    options = [a for a in accepted if a != ""] if bfcl else accepted
    errors = [_matches_alternative(pred, a) for a in options]
    if any(e is None for e in errors):
        return None
    return "wrong_value" if "wrong_value" in errors else "wrong_type"


# --- calls -----------------------------------------------------------------------------

# (name, {arg: [accepted values]}, whether "" means "may be omitted" as in BFCL)
Gold = tuple[str, dict[str, list[Any]], bool]


def _gold_calls(example: Example) -> list[Gold]:
    if example.accept is not None:
        return [(*next(iter(g.items())), True) for g in example.accept]
    return [(c.name, {k: [v] for k, v in c.arguments.items()}, False) for c in example.calls]


def _call_error(pred: Call, gold: Gold, tools: dict[str, dict[str, Any]]) -> str | None:
    name, accepted, bfcl = gold
    if pred.name != name:
        return "wrong_tool"
    props = tools.get(name, {}).get("parameters", {}).get("properties", {})
    found = set()
    for arg, value in pred.arguments.items():
        if arg not in accepted:
            if arg not in props:
                found.add("hallucinated_arg")
            elif "default" not in props[arg] or values_equal(value, props[arg]["default"]):
                found.add("extra_arg")
            continue
        err = _value_error(value, accepted[arg], bfcl)
        if err:
            found.add(err)
    for arg, options in accepted.items():
        if arg not in pred.arguments and not (bfcl and "" in options):
            found.add("missing_arg")
    return next((e for e in ARG_ERRORS if e in found), None)


def _max_matching(ok: list[list[bool]]) -> int:
    """Size of a maximum matching in a bipartite graph given as an adjacency matrix
    (rows = gold, cols = pred). Kuhn's algorithm; n is tiny."""
    match_col: dict[int, int] = {}

    def try_row(r: int, seen: set[int]) -> bool:
        for c, edge in enumerate(ok[r]):
            if edge and c not in seen:
                seen.add(c)
                if c not in match_col or try_row(match_col[c], seen):
                    match_col[c] = r
                    return True
        return False

    return sum(try_row(r, set()) for r in range(len(ok)))


def score_example(pred: list[Call] | None, example: Example) -> Score:
    if pred is None:
        return Score(False, False, False, "invalid_json")
    gold = _gold_calls(example)
    if not gold:
        return Score(not pred, True, not pred, None if not pred else "over_call")
    if not pred:
        return Score(False, True, False, "missed_call")

    name_match = sorted(c.name for c in pred) == sorted(g[0] for g in gold)
    if len(pred) < len(gold) and len(gold) > 1:
        return Score(False, True, name_match, "missed_multi_call")
    if len(pred) > len(gold):
        return Score(False, True, name_match, "extra_call")
    if not name_match:
        return Score(False, True, False, "wrong_tool")

    tools = {t["name"]: t for t in example.tools}
    errors = [[_call_error(p, g, tools) for p in pred] for g in gold]
    if _max_matching([[e is None for e in row] for row in errors]) == len(gold):
        return Score(True, True, True, None)
    # Report the most specific argument error among same-name pairs.
    seen = {e for row in errors for e in row if e and e != "wrong_tool"}
    return Score(False, True, True, next((e for e in ARG_ERRORS if e in seen), "wrong_value"))

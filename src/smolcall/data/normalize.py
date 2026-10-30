"""Parse each source dataset into :class:`smolcall.types.Example`.

Every parser returns ``None`` for rows it can't trust (unparseable arguments, a call to a
function that isn't in the offered tools, no tools at all) rather than guessing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from smolcall.types import Call, Example

_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "int": "integer"}


def normalize_schema(schema: Any) -> Any:
    """Map Python-style type names (BFCL uses ``dict``, ``float``, ``tuple``, ``any``) to
    JSON Schema, recursively."""
    if isinstance(schema, list):
        return [normalize_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            if value == "any":
                continue
            out[key] = _TYPE_MAP.get(value, value)
        elif key in ("properties",) and isinstance(value, dict):
            out[key] = {k: normalize_schema(v) for k, v in value.items()}
        elif key in ("items", "additionalProperties", "anyOf", "oneOf"):
            out[key] = normalize_schema(value)
        else:
            out[key] = value
    return out


def _tool(name: str, description: str, parameters: Any) -> dict[str, Any]:
    params = normalize_schema(parameters) if isinstance(parameters, dict) else {}
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    return {"name": name, "description": description or "", "parameters": params}


def _valid_calls(calls: list[Call], tools: list[dict[str, Any]]) -> bool:
    """Calls must name an offered tool. Empty-string argument values are rejected too: in
    glaive they are sloppy labels (``"album": ""``) and in hermes placeholders for the
    output of an earlier call, and neither is something a model should learn to emit."""
    names = {t["name"] for t in tools}
    return all(
        c.name in names
        and isinstance(c.arguments, dict)
        and all(v != "" for v in c.arguments.values())
        for c in calls
    )


# --- glaive-function-calling-v2 -------------------------------------------------------

_GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s")
_GLAIVE_CALL = re.compile(
    r"<functioncall>\s*\{\s*\"name\":\s*\"(?P<name>[^\"]+)\",\s*\"arguments\":\s*'(?P<args>.*)'\s*\}",
    re.DOTALL,
)


def _glaive_tools(system: str) -> list[dict[str, Any]]:
    marker = "Use them if required -"
    if marker not in system:
        return []
    text = system.split(marker, 1)[1]
    decoder = json.JSONDecoder()
    tools, i = [], 0
    while True:
        start = text.find("{", i)
        if start < 0:
            break
        obj, i = decoder.raw_decode(text, start)
        if isinstance(obj, dict) and "name" in obj:
            tools.append(_tool(obj["name"], obj.get("description", ""), obj.get("parameters")))
    return tools


def _glaive_turns(chat: str) -> list[tuple[str, str]]:
    parts = _GLAIVE_TURN.split(chat)
    # split() with one group gives ['', role, text, role, text, ...]
    turns = []
    for role, text in zip(parts[1::2], parts[2::2], strict=True):
        turns.append((role, text.replace("<|endoftext|>", "").strip()))
    return turns


def parse_glaive(row: dict[str, str], index: int) -> Example | None:
    """First user turn of a glaive conversation. The target is the call the assistant makes
    before the next user message or function response, or no call if it only replies."""
    try:
        tools = _glaive_tools(row["system"])
    except json.JSONDecodeError:
        return None
    if not tools:
        return None
    turns = _glaive_turns(row["chat"])
    if not turns or turns[0][0] != "USER" or not turns[0][1]:
        return None

    replies = []
    for role, text in turns[1:]:
        if role != "ASSISTANT":
            break
        replies.append(text)
    if not replies:
        return None

    calls: list[Call] = []
    for text in replies:
        m = _GLAIVE_CALL.search(text)
        if m:
            try:
                args = json.loads(m.group("args"))
            except json.JSONDecodeError:
                return None
            calls = [Call(m.group("name"), args)]
            break
    if not _valid_calls(calls, tools):
        return None
    return Example(f"glaive-{index}", "glaive", turns[0][1], tools, calls)


# --- hermes-function-calling-v1 (func-calling-singleturn) ------------------------------

_HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_hermes(row: dict[str, Any]) -> Example | None:
    try:
        raw_tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    except json.JSONDecodeError:
        return None
    tools = []
    for t in raw_tools:
        fn = t.get("function", t)
        tools.append(_tool(fn["name"], fn.get("description", ""), fn.get("parameters")))

    convo = {c["from"]: c["value"] for c in row["conversations"]}
    query, answer = convo.get("human", "").strip(), convo.get("gpt")
    if not tools or not query or answer is None:
        return None
    calls = []
    for block in _HERMES_CALL.findall(answer):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            return None
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        calls.append(Call(obj.get("name", ""), args))
    if not _valid_calls(calls, tools):
        return None
    return Example(f"hermes-{row['id']}", "hermes", query, tools, calls)


# --- Berkeley Function-Calling Leaderboard (eval only) ---------------------------------


def _bfcl_name(name: str, offered: set[str]) -> str:
    """A few BFCL answers drop the namespace (``find_closest`` for the offered
    ``restaurant_search.find_closest``); map them to the one offered tool they name."""
    if name in offered:
        return name
    matches = [n for n in offered if n.endswith("." + name)]
    return matches[0] if len(matches) == 1 else name


def parse_bfcl(question: dict[str, Any], answer: dict[str, Any] | None, category: str) -> Example:
    turn = question["question"][0]
    query = next(m["content"] for m in reversed(turn) if m["role"] == "user")
    tools = [
        _tool(f["name"], f.get("description", ""), f.get("parameters"))
        for f in question["function"]
    ]
    accept: list[dict[str, dict[str, list[Any]]]] = answer["ground_truth"] if answer else []
    names = {t["name"] for t in tools}
    accept = [{_bfcl_name(n, names): alts for n, alts in g.items()} for g in accept]
    calls = []
    for gold in accept:
        ((name, alts),) = gold.items()
        args = {k: v[0] for k, v in alts.items() if "" not in v}
        calls.append(Call(name, args))
    return Example(
        f"bfcl-{question['id']}",
        "bfcl",
        query,
        tools,
        calls,
        accept=accept,
        meta={"bfcl_category": category},
    )

"""The one prompt every local system sees, for training and for evaluation.

The model gets the tools as JSON Schema in the system message and must answer with a bare
JSON array of calls, or ``[]`` when no tool fits. Frontier models get the same instruction
text but receive the tools through their native tool-use API (see ``eval/frontier.py``).
"""

from __future__ import annotations

import json
from typing import Any

from smolcall.types import Call, Example

INSTRUCTION = (
    "You are a function-calling assistant. Decide which of the available tools to call "
    "for the user's request, and with which arguments.\n"
    "- Call a tool only if it fits the request and the request gives every required "
    "argument. Never invent argument values the user did not give or clearly imply.\n"
    "- Use several calls when the request asks for several things.\n"
    "- If no tool fits or required information is missing, call nothing."
)

FORMAT = (
    "Reply with only a JSON array of calls, no prose: "
    '[{"name": "<tool name>", "arguments": {"<arg>": <value>}}]. '
    "Reply with [] to call nothing."
)


def render_tools(tools: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)


def render_calls(calls: list[Call]) -> str:
    return json.dumps(
        [{"name": c.name, "arguments": c.arguments} for c in calls], ensure_ascii=False
    )


def system_prompt(tools: list[dict[str, Any]], shots: list[Example] | None = None) -> str:
    parts = [INSTRUCTION, FORMAT]
    if shots:
        demo = ["Examples (these use other tools than yours):"]
        for i, s in enumerate(shots, 1):
            demo += [
                f"Example {i} tools:\n{render_tools(s.tools)}",
                f"Example {i} request: {s.query}",
                f"Example {i} answer: {render_calls(s.calls)}",
            ]
        parts.append("\n\n".join(demo))
    parts.append(f"Available tools:\n{render_tools(tools)}")
    return "\n\n".join(parts)


def messages(example: Example, shots: list[Example] | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(example.tools, shots)},
        {"role": "user", "content": example.query},
    ]


def training_record(example: Example) -> dict[str, Any]:
    """One line of the mlx_lm ``chat`` JSONL format."""
    return {
        "messages": [
            *messages(example),
            {"role": "assistant", "content": render_calls(example.calls)},
        ]
    }


def pick_shots(train: list[Example]) -> list[Example]:
    """Three fixed demonstrations: one single call, one multi-call, one "call nothing".

    Chosen deterministically as the shortest train example of each kind with at most three
    tools, so the few-shot prompt stays small and never changes between runs.
    """
    out = []
    for cat in ("single", "multi", "none"):
        pool = [e for e in train if e.category == cat and len(e.tools) <= 3]
        out.append(min(pool, key=lambda e: (len(system_prompt(e.tools)) + len(e.query), e.id)))
    return out

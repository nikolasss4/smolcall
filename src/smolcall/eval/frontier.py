"""Frontier baselines through Anthropic's native tool-use API.

The model gets the same instruction text as the local models (``prompt.INSTRUCTION``) and
the same tool schemas, passed as API tools instead of text. Its ``tool_use`` blocks are the
predicted calls; a reply with no ``tool_use`` block means it called nothing. Settings are
the API defaults for each model (no explicit thinking or effort), with automatic retries.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from smolcall.eval.systems import Prediction
from smolcall.prompt import INSTRUCTION
from smolcall.types import Call, Example

# USD per million tokens, Anthropic first-party API (pricing as of 2026-10-06).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

_BAD_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def api_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert tools to API format. Tool names may only use ``[a-zA-Z0-9_-]`` (BFCL has
    dotted names like ``math.factorial``), so names are sanitized and mapped back."""
    out, back = [], {}
    for t in tools:
        safe = _BAD_NAME_CHARS.sub("_", t["name"])[:128]
        while safe in back:
            safe = safe[:120] + "_"
        back[safe] = t["name"]
        schema = dict(t.get("parameters") or {})
        schema["type"] = "object"
        schema.setdefault("properties", {})
        out.append({"name": safe, "description": t.get("description", ""), "input_schema": schema})
    return out, back


def calls_from_content(content: list[Any], back: dict[str, str]) -> tuple[list[Call], str]:
    calls, texts = [], []
    for block in content:
        if block.type == "tool_use":
            calls.append(Call(back.get(block.name, block.name), dict(block.input)))
        elif block.type == "text":
            texts.append(block.text)
    return calls, "\n".join(texts)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


class AnthropicSystem:
    def __init__(self, name: str, model: str, client: Any = None, workers: int = 8) -> None:
        self.name = name
        self.model = model
        self.workers = workers
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=6)
        self.client = client

    def _one(self, ex: Example) -> Prediction:
        import anthropic

        tools, back = api_tools(ex.tools)
        start = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=INSTRUCTION,
                tools=tools,
                tool_choice={"type": "auto"},
                messages=[{"role": "user", "content": ex.query}],
            )
        except anthropic.BadRequestError as e:
            # Recorded as a failed prediction (counts against accuracy), not skipped.
            return Prediction(ex.id, f"API error: {e.message}", None, {"error": "bad_request"})
        latency = time.perf_counter() - start
        calls, text = calls_from_content(resp.content, back)
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "latency_s": round(latency, 3),
            "cost_usd": cost_usd(self.model, u.input_tokens, u.output_tokens),
            "stop_reason": resp.stop_reason,
        }
        raw = text or json.dumps([{"name": c.name, "arguments": c.arguments} for c in calls])
        return Prediction(ex.id, raw, calls, usage, native=True)

    def predict(self, examples: Sequence[Example]) -> Iterator[Prediction]:
        with ThreadPoolExecutor(self.workers) as pool:
            yield from pool.map(self._one, examples)

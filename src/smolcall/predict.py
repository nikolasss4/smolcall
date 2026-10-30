"""One request in, tool calls out: the smallest way to use the fine-tune.

python predict.py --tools examples/tools.json "What's the weather in Oslo in fahrenheit?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smolcall.eval.scorer import parse_output
from smolcall.prompt import messages
from smolcall.types import Example

DEFAULT_MODEL = "models/smolcall-4bit"


def predict(
    query: str, tools: list[dict[str, Any]], model: str = DEFAULT_MODEL, adapter: str | None = None
) -> tuple[list[dict[str, Any]] | None, str]:
    """Return ``(calls, raw_text)``. ``calls`` is ``[]`` for "call nothing" and ``None``
    if the model's output could not be parsed."""
    from mlx_lm import generate, load

    model_obj, tokenizer = load(model, adapter_path=adapter)
    prompt = tokenizer.apply_chat_template(
        messages(Example("cli", "cli", query, tools, [])), add_generation_prompt=True
    )
    text = generate(model_obj, tokenizer, prompt, max_tokens=768)
    calls = parse_output(text)
    out = None if calls is None else [{"name": c.name, "arguments": c.arguments} for c in calls]
    return out, text


def load_tools(path: Path) -> list[dict[str, Any]]:
    tools = json.loads(path.read_text())
    # Accept OpenAI-style {"type": "function", "function": {...}} entries as well.
    return [t.get("function", t) for t in tools]

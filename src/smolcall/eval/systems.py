"""Systems under evaluation. Each turns examples into raw text plus parsed calls.

All local systems share ``prompt.messages`` and greedy decoding. Frontier systems live in
``frontier.py``; ``ReplaySystem`` re-scores recorded outputs and is what CI runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from smolcall.eval.scorer import parse_output
from smolcall.prompt import messages
from smolcall.types import Call, Example


@dataclass
class Prediction:
    id: str
    text: str
    calls: list[Call] | None
    # Per-request numbers when the system measures them (frontier APIs do).
    usage: dict[str, Any] = field(default_factory=dict)
    # True when calls came from a native tool-use API rather than parsed text.
    native: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["calls"] = None if self.calls is None else [asdict(c) for c in self.calls]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prediction:
        calls = d.get("calls")
        return cls(
            id=d["id"],
            text=d["text"],
            calls=None if calls is None else [Call(c["name"], c["arguments"]) for c in calls],
            usage=d.get("usage", {}),
            native=d.get("native", False),
        )


class System(Protocol):
    name: str

    def predict(self, examples: Sequence[Example]) -> Iterator[Prediction]: ...


class MLXSystem:
    """A local mlx_lm model (optionally with a LoRA adapter), batched greedy decoding."""

    def __init__(
        self,
        name: str,
        model: str,
        adapter: str | None = None,
        shots: list[Example] | None = None,
        batch_size: int = 16,
        max_tokens: int = 768,
    ) -> None:
        self.name = name
        self.model_path = model
        self.adapter = adapter
        self.shots = shots
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._loaded: tuple[Any, Any] | None = None

    def load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from mlx_lm import load

            model, tokenizer = load(self.model_path, adapter_path=self.adapter)
            self._loaded = (model, tokenizer)
        return self._loaded

    def prompt_tokens(self, example: Example) -> list[int]:
        _, tokenizer = self.load()
        tokens: list[int] = tokenizer.apply_chat_template(
            messages(example, self.shots), add_generation_prompt=True
        )
        return tokens

    def predict(self, examples: Sequence[Example]) -> Iterator[Prediction]:
        from mlx_lm import batch_generate

        model, tokenizer = self.load()
        prompts = [(e, self.prompt_tokens(e)) for e in examples]
        # Batch prompts of similar length together to waste less work on padding.
        prompts.sort(key=lambda p: len(p[1]))
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            start = time.perf_counter()
            out = batch_generate(
                model, tokenizer, [p for _, p in chunk], max_tokens=self.max_tokens
            )
            per_example = (time.perf_counter() - start) / len(chunk)
            for (ex, prompt), text in zip(chunk, out.texts, strict=True):
                yield Prediction(
                    ex.id,
                    text,
                    parse_output(text),
                    usage={"input_tokens": len(prompt), "batch_seconds_per_example": per_example},
                )


class ReplaySystem:
    """Recorded predictions, re-parsed from their raw text so scorer changes apply."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.records = {
            r["id"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line)
        }

    def predict(self, examples: Sequence[Example]) -> Iterator[Prediction]:
        for ex in examples:
            p = Prediction.from_dict(self.records[ex.id])
            if not p.native:
                p.calls = parse_output(p.text)
            yield p

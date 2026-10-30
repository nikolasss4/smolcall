"""The one record format every dataset is normalized into.

An example is a single user request, the tools offered with it, and the gold tool calls.
An empty ``calls`` list means the right answer is to call nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Category = Literal["none", "single", "multi"]


@dataclass
class Call:
    name: str
    arguments: dict[str, Any]


@dataclass
class Example:
    id: str
    source: str
    query: str
    tools: list[dict[str, Any]]
    calls: list[Call]
    # BFCL lists several acceptable values per argument. When present, the scorer uses
    # this instead of ``calls``: one dict per gold call, ``{name: {arg: [accepted values]}}``.
    # An empty string among the accepted values means the argument may be omitted.
    accept: list[dict[str, dict[str, list[Any]]]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> Category:
        if not self.calls:
            return "none"
        return "single" if len(self.calls) == 1 else "multi"

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["accept"] is None:
            del d["accept"]
        if not d["meta"]:
            del d["meta"]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            id=d["id"],
            source=d["source"],
            query=d["query"],
            tools=d["tools"],
            calls=[Call(c["name"], c["arguments"]) for c in d["calls"]],
            accept=d.get("accept"),
            meta=d.get("meta", {}),
        )

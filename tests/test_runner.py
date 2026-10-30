import json
from pathlib import Path

from smolcall.eval.runner import summarize
from smolcall.eval.systems import Prediction, ReplaySystem
from smolcall.types import Call, Example

TOOLS = [{"name": "f", "description": "", "parameters": {"type": "object", "properties": {}}}]
EXAMPLES = [
    Example("a", "t", "q1", TOOLS, [Call("f", {"x": 1})]),
    Example("b", "t", "q2", TOOLS, []),
]


def test_replay_reparses_text_but_keeps_native_calls(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    records = [
        # Raw text is re-parsed, so the stored "calls" is ignored for text systems.
        Prediction("a", '[{"name": "f", "arguments": {"x": 1}}]', None).to_dict(),
        Prediction("b", "ignored", [Call("f", {})], {"cost_usd": 0.1}, native=True).to_dict(),
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    preds = {p.id: p for p in ReplaySystem("x", path).predict(EXAMPLES)}
    assert preds["a"].calls == [Call("f", {"x": 1})]
    assert preds["b"].calls == [Call("f", {})]


def test_summarize_reports_cost_only_when_every_prediction_has_it() -> None:
    usage = {"cost_usd": 0.002, "latency_s": 1.0, "output_tokens": 10}
    preds = {
        "a": Prediction("a", "", [Call("f", {"x": 1})], usage, native=True),
        "b": Prediction("b", "", [], usage, native=True),
    }
    m = summarize("sys", "test", EXAMPLES, preds)
    assert m["accuracy"]["value"] == 1.0
    assert m["cost_usd_per_1k_calls"] == 2.0
    preds["b"].usage = {}
    assert "cost_usd_per_1k_calls" not in summarize("sys", "test", EXAMPLES, preds)

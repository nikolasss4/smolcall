import json

from smolcall.prompt import messages, pick_shots, render_calls, system_prompt, training_record
from smolcall.types import Call, Example

TOOLS = [{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def _ex(i: int, n_calls: int, query: str = "q") -> Example:
    return Example(f"e{i}", "t", query, TOOLS, [Call("f", {"x": i})] * n_calls)


def test_training_record_targets_compact_json_calls() -> None:
    rec = training_record(_ex(1, 2))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert json.loads(rec["messages"][2]["content"]) == [
        {"name": "f", "arguments": {"x": 1}},
        {"name": "f", "arguments": {"x": 1}},
    ]
    assert render_calls([]) == "[]"


def test_system_prompt_lists_tools_and_optional_shots() -> None:
    plain = system_prompt(TOOLS)
    assert '"name": "f"' in plain and "Example 1" not in plain
    shots = [_ex(1, 1, "shot query")]
    assert "shot query" in system_prompt(TOOLS, shots)
    assert messages(_ex(2, 0), shots)[1] == {"role": "user", "content": "q"}


def test_pick_shots_is_one_of_each_kind_and_deterministic() -> None:
    train = [_ex(i, i % 3, "x" * (10 + i)) for i in range(12)]
    shots = pick_shots(train)
    assert [s.category for s in shots] == ["single", "multi", "none"]
    assert [s.id for s in shots] == [s.id for s in pick_shots(list(reversed(train)))]

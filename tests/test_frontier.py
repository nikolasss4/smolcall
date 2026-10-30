from types import SimpleNamespace
from typing import Any

from smolcall.eval.frontier import AnthropicSystem, api_tools, calls_from_content, cost_usd
from smolcall.types import Example


def test_api_tools_sanitizes_names_and_maps_back() -> None:
    tools = [
        {
            "name": "math.factorial",
            "description": "n!",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
        },
        {"name": "math_factorial", "description": "", "parameters": {}},
    ]
    out, back = api_tools(tools)
    names = [t["name"] for t in out]
    assert names[0] == "math_factorial" and names[1] != names[0]
    assert back[names[0]] == "math.factorial" and back[names[1]] == "math_factorial"
    assert out[1]["input_schema"] == {"type": "object", "properties": {}}


def test_calls_from_content() -> None:
    content = [
        SimpleNamespace(type="text", text="Calling."),
        SimpleNamespace(type="tool_use", name="math_factorial", input={"n": 5}),
    ]
    calls, text = calls_from_content(content, {"math_factorial": "math.factorial"})
    assert [(c.name, c.arguments) for c in calls] == [("math.factorial", {"n": 5})]
    assert text == "Calling."


def test_cost() -> None:
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == 0.60


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="f", input={"a": 1})],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            stop_reason="tool_use",
        )


def test_system_uses_native_tools_and_records_usage() -> None:
    client = FakeClient()
    system = AnthropicSystem("claude-haiku-5-5", "claude-haiku-5-5", client=client, workers=1)
    ex = Example("e1", "t", "do f", [{"name": "f", "description": "", "parameters": {}}], [])
    (pred,) = list(system.predict([ex]))
    assert pred.native and [c.name for c in pred.calls or []] == ["f"]
    assert pred.usage["cost_usd"] == cost_usd("claude-haiku-5-5", 100, 20)
    req = client.requests[0]
    assert req["tool_choice"] == {"type": "auto"} and req["tools"][0]["name"] == "f"
    assert req["messages"] == [{"role": "user", "content": "do f"}]

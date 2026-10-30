import json
from pathlib import Path

from smolcall.predict import load_tools


def test_load_tools_accepts_plain_and_openai_style(tmp_path: Path) -> None:
    plain = {"name": "f", "description": "", "parameters": {"type": "object", "properties": {}}}
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([plain, {"type": "function", "function": {**plain, "name": "g"}}]))
    assert [t["name"] for t in load_tools(path)] == ["f", "g"]


def test_example_tools_file_is_valid() -> None:
    tools = load_tools(Path(__file__).resolve().parents[1] / "examples" / "tools.json")
    assert {t["name"] for t in tools} == {"get_weather", "convert_currency"}

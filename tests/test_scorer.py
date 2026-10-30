import pytest

from smolcall.eval.scorer import parse_output, score_example, values_equal
from smolcall.types import Call, Example

TOOLS = [
    {
        "name": "get_weather",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "default": 1},
                "units": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_time",
        "description": "",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}},
    },
]


def example(*calls: Call, accept: list | None = None) -> Example:
    return Example("e", "t", "q", TOOLS, list(calls), accept=accept)


# --- parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '[{"name": "get_time", "arguments": {"tz": "UTC"}}]',
        '```json\n[{"name": "get_time", "arguments": {"tz": "UTC"}}]\n```',
        'Sure! [{"name": "get_time", "arguments": {"tz": "UTC"}}] Done.',
        '{"name": "get_time", "arguments": {"tz": "UTC"}}',
        '{"tool_calls": [{"name": "get_time", "arguments": "{\\"tz\\": \\"UTC\\"}"}]}',
    ],
)
def test_parse_output_accepts_common_shapes(text: str) -> None:
    calls = parse_output(text)
    assert calls is not None
    assert [(c.name, c.arguments) for c in calls] == [("get_time", {"tz": "UTC"})]


def test_parse_output_empty_list_means_no_call() -> None:
    assert parse_output("[]") == []
    assert parse_output("  [ ]\n") == []


@pytest.mark.parametrize(
    "text",
    [
        "{oops}",
        '[{"name": "x", "arguments": {"a": 1}',
        '[{"arguments": {}}]',
        "[1, 2]",
    ],
)
def test_parse_output_rejects_invalid(text: str) -> None:
    assert parse_output(text) is None


def test_prose_without_json_counts_as_no_call() -> None:
    assert parse_output("Which city do you mean?") == []


def test_parse_output_accepts_one_object_per_line() -> None:
    text = '{"name": "a", "arguments": {}}\n{"name": "b", "arguments": {"x": 1}}'
    assert parse_output(text) == [Call("a", {}), Call("b", {"x": 1})]


@pytest.mark.parametrize(
    "text",
    [
        '[{"name": "a", "arguments": {}}], [{"name": "b", "arguments": {"x": 1}}]',
        '[{"name": "a", "arguments": {}}]\n[{"name": "b", "arguments": {"x": 1}}]',
    ],
)
def test_parse_output_accepts_one_array_per_call(text: str) -> None:
    assert parse_output(text) == [Call("a", {}), Call("b", {"x": 1})]


def test_parse_output_reads_thinking_free_tail() -> None:
    text = '<think>maybe get_time</think>\n[{"name": "get_time", "arguments": {}}]'
    assert parse_output(text) == [Call("get_time", {})]


# --- value equality ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred", "gold", "verdict"),
    [
        (5, 5.0, None),
        (5, 6, "wrong_value"),
        ("5", 5, "wrong_type"),
        (True, 1, "wrong_type"),
        ("New York", "new york ", None),
        ("NYC", "New York", "wrong_value"),
        ([1, 2], [1, 2], None),
        ([2, 1], [1, 2], "wrong_value"),
        ({"a": 1, "b": "X"}, {"b": "x", "a": 1.0}, None),
        ({"a": 1}, {"a": 1, "b": 2}, "wrong_value"),
        (None, None, None),
    ],
)
def test_values_equal(pred: object, gold: object, verdict: str | None) -> None:
    assert values_equal(pred, gold) == verdict


# --- whole-example scoring --------------------------------------------------------------


def test_exact_single_call_is_correct_regardless_of_arg_order() -> None:
    s = score_example(
        [Call("get_weather", {"units": "c", "city": "Paris"})],
        example(Call("get_weather", {"city": "Paris", "units": "c"})),
    )
    assert s.correct and s.json_valid and s.name_match and s.error is None


def test_invalid_json() -> None:
    s = score_example(None, example(Call("get_time", {})))
    assert not s.correct and not s.json_valid and s.error == "invalid_json"


def test_no_call_cases() -> None:
    assert score_example([], example()).correct
    over = score_example([Call("get_time", {})], example())
    assert over.error == "over_call" and not over.correct
    missed = score_example([], example(Call("get_time", {})))
    assert missed.error == "missed_call"


def test_wrong_tool() -> None:
    s = score_example([Call("get_time", {})], example(Call("get_weather", {"city": "Rome"})))
    assert s.error == "wrong_tool" and not s.name_match


@pytest.mark.parametrize(
    ("pred_args", "error"),
    [
        ({}, "missing_arg"),
        ({"city": "Rome", "country": "IT"}, "hallucinated_arg"),
        ({"city": "Rome", "units": "f"}, "extra_arg"),
        ({"city": 7}, "wrong_type"),
        ({"city": "Milan"}, "wrong_value"),
    ],
)
def test_argument_errors(pred_args: dict, error: str) -> None:
    s = score_example(
        [Call("get_weather", pred_args)], example(Call("get_weather", {"city": "Rome"}))
    )
    assert s.name_match and not s.correct and s.error == error


def test_extra_arg_equal_to_schema_default_is_allowed() -> None:
    s = score_example(
        [Call("get_weather", {"city": "Rome", "days": 1})],
        example(Call("get_weather", {"city": "Rome"})),
    )
    assert s.correct


def test_multi_call_is_order_insensitive() -> None:
    gold = example(Call("get_weather", {"city": "Rome"}), Call("get_weather", {"city": "Oslo"}))
    pred = [Call("get_weather", {"city": "Oslo"}), Call("get_weather", {"city": "Rome"})]
    assert score_example(pred, gold).correct


def test_multi_call_count_errors() -> None:
    gold = example(Call("get_weather", {"city": "Rome"}), Call("get_time", {"tz": "CET"}))
    fewer = score_example([Call("get_weather", {"city": "Rome"})], gold)
    assert fewer.error == "missed_multi_call"
    more = score_example(
        [Call("get_time", {"tz": "CET"})] * 2 + [Call("get_weather", {"city": "Rome"})], gold
    )
    assert more.error == "extra_call"


def test_multi_call_value_error_is_reported() -> None:
    gold = example(Call("get_weather", {"city": "Rome"}), Call("get_time", {"tz": "CET"}))
    s = score_example(
        [Call("get_weather", {"city": "Rome"}), Call("get_time", {"tz": "UTC"})], gold
    )
    assert s.name_match and s.error == "wrong_value"


def test_bfcl_alternatives_and_optional_args() -> None:
    accept = [{"get_weather": {"city": ["New York", "NYC"], "units": ["", "f"]}}]
    ex = example(Call("get_weather", {"city": "New York"}), accept=accept)
    assert score_example([Call("get_weather", {"city": "NYC"})], ex).correct
    assert score_example([Call("get_weather", {"city": "NYC", "units": "f"})], ex).correct
    bad = score_example([Call("get_weather", {"city": "NYC", "units": "c"})], ex)
    assert bad.error == "wrong_value"


def test_bfcl_nested_dict_alternatives() -> None:
    accept = [{"get_time": {"tz": [{"region": ["Europe"], "city": ["Paris", "paris"]}]}}]
    ex = example(Call("get_time", {}), accept=accept)
    assert score_example(
        [Call("get_time", {"tz": {"region": "Europe", "city": "Paris"}})], ex
    ).correct


def test_empty_string_is_a_real_value_outside_bfcl() -> None:
    ex = example(Call("get_time", {"tz": ""}))
    assert score_example([Call("get_time", {"tz": ""})], ex).correct
    assert score_example([Call("get_time", {})], ex).error == "missing_arg"

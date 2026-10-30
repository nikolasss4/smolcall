from smolcall.data.normalize import (
    normalize_schema,
    parse_bfcl,
    parse_glaive,
    parse_hermes,
)

GLAIVE_SYSTEM = (
    "SYSTEM: You are a helpful assistant with access to the following functions. "
    "Use them if required -\n"
    '{\n    "name": "get_rate",\n    "description": "Exchange rate",\n'
    '    "parameters": {"type": "object", "properties": {"base": {"type": "string"}},'
    ' "required": ["base"]}\n}\n\n'
    '{\n    "name": "convert",\n    "description": "Convert",\n'
    '    "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}}\n}\n'
)


def test_glaive_call_with_single_quoted_arguments() -> None:
    chat = (
        "USER: What is the USD rate?\n\n\n"
        'ASSISTANT: <functioncall> {"name": "get_rate", "arguments": \'{"base": "USD"}\'}'
        " <|endoftext|>\n\n\nFUNCTION RESPONSE: {}\n\n\nASSISTANT: It is 1. <|endoftext|>\n"
    )
    ex = parse_glaive({"system": GLAIVE_SYSTEM, "chat": chat}, 7)
    assert ex is not None
    assert ex.id == "glaive-7"
    assert ex.query == "What is the USD rate?"
    assert ex.tool_names == ["get_rate", "convert"]
    assert [(c.name, c.arguments) for c in ex.calls] == [("get_rate", {"base": "USD"})]


def test_glaive_text_reply_is_a_no_call_example() -> None:
    chat = (
        "USER: Book me a flight.\n\n\n"
        "ASSISTANT: I'm sorry, I can't book flights. <|endoftext|>\n\n\n"
        'USER: ok\n\n\nASSISTANT: <functioncall> {"name": "get_rate", '
        '"arguments": \'{"base": "EUR"}\'} <|endoftext|>\n'
    )
    ex = parse_glaive({"system": GLAIVE_SYSTEM, "chat": chat}, 1)
    assert ex is not None
    assert ex.calls == []
    assert ex.category == "none"


def test_glaive_preamble_then_call_counts_as_call() -> None:
    chat = (
        "USER: Convert 5 dollars\n\n\nASSISTANT: Sure, one moment. <|endoftext|>\n\n\n"
        'ASSISTANT: <functioncall> {"name": "convert", "arguments": \'{"amount": 5}\'}'
        " <|endoftext|>\n"
    )
    ex = parse_glaive({"system": GLAIVE_SYSTEM, "chat": chat}, 2)
    assert ex is not None
    assert ex.calls[0].arguments == {"amount": 5}


def test_glaive_rejects_unknown_function_and_missing_tools() -> None:
    bad_call = (
        'USER: hi\n\n\nASSISTANT: <functioncall> {"name": "nope", "arguments": \'{}\'}'
        " <|endoftext|>\n"
    )
    assert parse_glaive({"system": GLAIVE_SYSTEM, "chat": bad_call}, 3) is None
    no_tools = "SYSTEM: You are a helpful assistant, with no access to external functions.\n"
    assert parse_glaive({"system": no_tools, "chat": "USER: hi\n\n\nASSISTANT: hello\n"}, 4) is None


def test_glaive_rejects_unparseable_arguments() -> None:
    chat = (
        'USER: rate?\n\n\nASSISTANT: <functioncall> {"name": "get_rate", '
        '"arguments": \'{"base": USD}\'} <|endoftext|>\n'
    )
    assert parse_glaive({"system": GLAIVE_SYSTEM, "chat": chat}, 5) is None


def test_hermes_parallel_calls() -> None:
    tools = (
        '[{"type": "function", "function": {"name": "a", "description": "A", '
        '"parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}},'
        ' {"type": "function", "function": {"name": "b", "description": "B", '
        '"parameters": {"type": "object", "properties": {}}}}]'
    )
    row = {
        "id": "abc",
        "tools": tools,
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model."},
            {"from": "human", "value": "Do a and b"},
            {
                "from": "gpt",
                "value": '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>\n'
                '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>\n',
            },
        ],
    }
    ex = parse_hermes(row)
    assert ex is not None
    assert ex.id == "hermes-abc"
    assert ex.category == "multi"
    assert [c.name for c in ex.calls] == ["a", "b"]
    assert ex.tools[0] == {
        "name": "a",
        "description": "A",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }


def test_normalize_schema_maps_python_style_types() -> None:
    schema = {
        "type": "dict",
        "properties": {
            "f": {"type": "float"},
            "t": {"type": "tuple", "items": {"type": "dict", "properties": {}}},
            "a": {"type": "any", "description": "anything"},
            "i": {"type": "integer"},
        },
    }
    assert normalize_schema(schema) == {
        "type": "object",
        "properties": {
            "f": {"type": "number"},
            "t": {"type": "array", "items": {"type": "object", "properties": {}}},
            "a": {"description": "anything"},
            "i": {"type": "integer"},
        },
    }


def test_bfcl_with_alternatives_and_irrelevance() -> None:
    q = {
        "id": "simple_1",
        "question": [[{"role": "user", "content": "Area of a circle r=5"}]],
        "function": [
            {
                "name": "circle.area",
                "description": "Area",
                "parameters": {
                    "type": "dict",
                    "properties": {"radius": {"type": "float"}, "units": {"type": "string"}},
                    "required": ["radius"],
                },
            }
        ],
    }
    answer = {
        "id": "simple_1",
        "ground_truth": [{"circle.area": {"radius": [5.0], "units": ["", "cm"]}}],
    }
    ex = parse_bfcl(q, answer, "simple")
    assert ex.id == "bfcl-simple_1"
    assert ex.meta == {"bfcl_category": "simple"}
    assert ex.tools[0]["parameters"]["type"] == "object"
    assert ex.accept == [{"circle.area": {"radius": [5.0], "units": ["", "cm"]}}]
    # The exact target picks the first non-omitted alternative and drops optional args.
    assert [(c.name, c.arguments) for c in ex.calls] == [("circle.area", {"radius": 5.0})]

    irr = parse_bfcl({**q, "id": "irrelevance_0"}, None, "irrelevance")
    assert irr.calls == []
    assert irr.accept == []


def test_glaive_rejects_empty_string_argument_values() -> None:
    chat = (
        'USER: rate?\n\n\nASSISTANT: <functioncall> {"name": "get_rate", '
        '"arguments": \'{"base": ""}\'} <|endoftext|>\n'
    )
    assert parse_glaive({"system": GLAIVE_SYSTEM, "chat": chat}, 6) is None


def test_bfcl_answer_without_namespace_maps_to_offered_tool() -> None:
    q = {
        "id": "simple_363",
        "question": [[{"role": "user", "content": "sushi in Boston"}]],
        "function": [{"name": "restaurant_search.find_closest", "parameters": {"type": "dict"}}],
    }
    answer = {"id": "simple_363", "ground_truth": [{"find_closest": {"location": ["Boston"]}}]}
    ex = parse_bfcl(q, answer, "simple")
    assert ex.accept == [{"restaurant_search.find_closest": {"location": ["Boston"]}}]
    assert ex.calls[0].name == "restaurant_search.find_closest"

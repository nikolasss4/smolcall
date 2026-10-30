from smolcall.data.split import (
    cap_per_toolset,
    check_leakage,
    dedup,
    normalize_query,
    split,
    tool_groups,
)
from smolcall.types import Call, Example


def ex(i: int, query: str, *tools: str) -> Example:
    return Example(
        id=f"x-{i}",
        source="t",
        query=query,
        tools=[{"name": t, "description": "", "parameters": {}} for t in tools],
        calls=[Call(tools[0], {})],
    )


def make_corpus(n: int = 400) -> list[Example]:
    # 100 small groups of two tools that are offered together, plus one big hub group.
    small = [ex(i, f"request number {i}", f"t{i % 100}_a", f"t{i % 100}_b") for i in range(n)]
    hub = [ex(n + i, f"hub request {i}", "hub", f"hub_{i % 30}") for i in range(200)]
    return small + hub


def test_normalize_query() -> None:
    assert normalize_query("  Hello   World!! ") == "hello world"
    assert normalize_query("What's my BMI?") == normalize_query("what's my  bmi")


def test_dedup_keeps_first_by_normalized_query() -> None:
    out = dedup([ex(1, "Hi there", "a"), ex(2, "hi  there!", "b"), ex(3, "other", "a")])
    assert [e.id for e in out] == ["x-1", "x-3"]


def test_split_holds_out_whole_tools_and_is_clean() -> None:
    splits = split(make_corpus(), test_group_frac=0.3, max_test_group=50, val_frac=0.1)
    assert splits["test"] and splits["train"] and splits["val"]
    # The hub group is too big to hold out, and no example is dropped.
    assert all("hub" not in e.tool_names for e in splits["test"])
    assert sum(len(v) for v in splits.values()) == 600
    assert check_leakage(splits) == []
    test_tools = {n for e in splits["test"] for n in e.tool_names}
    train_tools = {n for e in splits["train"] for n in e.tool_names}
    assert not test_tools & train_tools


def test_tool_groups_links_co_offered_tools() -> None:
    groups = tool_groups([ex(1, "a", "x", "y"), ex(2, "b", "y", "z"), ex(3, "c", "w")])
    assert groups["x"] == groups["y"] == groups["z"] == "x"
    assert groups["w"] == "w"


def test_split_is_deterministic() -> None:
    a = split(make_corpus())
    b = split(make_corpus())
    assert {k: [e.id for e in v] for k, v in a.items()} == {
        k: [e.id for e in v] for k, v in b.items()
    }


def test_split_removes_cross_split_duplicate_queries() -> None:
    corpus = make_corpus() + [ex(1000, "Request number 3", "tool_3")]
    splits = split(corpus)
    assert sum(len(v) for v in splits.values()) == 600
    assert check_leakage(splits) == []


def test_leakage_checker_flags_every_kind_of_leak() -> None:
    splits = {
        "train": [ex(1, "same query", "shared_tool"), ex(2, "a", "x")],
        "test": [ex(3, "Same query!", "shared_tool"), ex(2, "b", "y")],
    }
    problems = check_leakage(splits)
    assert any("id x-2" in p for p in problems)
    assert any("same query" in p for p in problems)
    assert any("'shared_tool'" in p for p in problems)


def test_cap_per_toolset() -> None:
    corpus = [ex(i, f"q{i}", "a") for i in range(10)] + [
        ex(100 + i, f"r{i}", "b") for i in range(2)
    ]
    out = cap_per_toolset(corpus, 3, seed=1)
    assert len(out) == 5
    assert out == cap_per_toolset(corpus, 3, seed=1)

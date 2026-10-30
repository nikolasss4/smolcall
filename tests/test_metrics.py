import pytest

from smolcall.eval.metrics import aggregate, bootstrap_ci, error_table
from smolcall.eval.scorer import Score
from smolcall.types import Call, Example


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    vals = [1.0] * 70 + [0.0] * 30
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.7 < hi
    assert lo > 0.55 and hi < 0.85
    assert bootstrap_ci(vals) == (lo, hi)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def _ex(n_calls: int) -> Example:
    return Example("e", "t", "q", [], [Call("f", {})] * n_calls)


def test_aggregate_splits_by_category() -> None:
    examples = [_ex(0), _ex(0), _ex(1), _ex(1), _ex(2)]
    scores = [
        Score(True, True, True, None),
        Score(False, True, False, "over_call"),
        Score(True, True, True, None),
        Score(False, True, False, "missed_call"),
        Score(False, False, False, "invalid_json"),
    ]
    agg = aggregate(scores, examples)
    assert agg["n"] == 5
    assert agg["accuracy"]["value"] == pytest.approx(0.4)
    assert agg["json_valid"]["value"] == pytest.approx(0.8)
    assert agg["no_call_accuracy"]["value"] == 0.5
    assert agg["false_call_rate"]["value"] == 0.5
    assert agg["missed_call_rate"]["value"] == pytest.approx(1 / 3)
    assert agg["single_accuracy"]["value"] == 0.5
    assert agg["multi_accuracy"] == {"n": 1, "value": 0.0, "ci95": [0.0, 0.0]}
    assert error_table(scores) == {"invalid_json": 1, "over_call": 1, "missed_call": 1}


def test_aggregate_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError):
        aggregate([Score(True, True, True, None)], [])

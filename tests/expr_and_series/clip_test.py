from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import MultiOutputExpressionError
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_clip_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(
        lower_only=nw.col("a").clip(lower_bound=3),
        upper_only=nw.col("a").clip(upper_bound=4),
        both=nw.col("a").clip(3, 4),
    )
    expected = {
        "lower_only": [3, 3, 3, 3, 5],
        "upper_only": [1, 2, 3, -4, 4],
        "both": [3, 3, 3, 3, 4],
    }
    assert_equal_data(result, expected)


def test_clip_expr_expressified(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "modin_pyarrow" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/17682
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3, -4, 5], "lb": [3, 2, 1, 1, 1], "ub": [4, 4, 2, 2, 2]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").clip("lb", nw.col("ub") + 1))
    expected_dict = {"a": [3, 2, 3, 1, 3]}
    assert_equal_data(result, expected_dict)


def test_clip_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3, -4, 5]}), eager_only=True)
    result = {
        "lower_only": df["a"].clip(lower_bound=3),
        "upper_only": df["a"].clip(upper_bound=4),
        "both": df["a"].clip(3, 4),
    }

    expected = {
        "lower_only": [3, 3, 3, 3, 5],
        "upper_only": [1, 2, 3, -4, 4],
        "both": [3, 3, 3, 3, 4],
    }
    assert_equal_data(result, expected)


def test_clip_series_expressified(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "modin_pyarrow" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/17682
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3, -4, 5], "lb": [3, 2, 1, 1, 1], "ub": [4, 4, 2, 2, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].clip(df["lb"], df["ub"] + 1).to_frame()
    expected_dict = {"a": [3, 2, 3, 1, 3]}
    assert_equal_data(result, expected_dict)


def test_clip_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises(MultiOutputExpressionError):
        df.select(nw.col("a").clip(nw.all(), nw.col("a", "b")))

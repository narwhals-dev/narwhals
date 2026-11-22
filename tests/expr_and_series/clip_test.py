from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import MultiOutputExpressionError
from tests.utils import Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (3, 4, [3, 3, 3, 3, 4]),
        (0, 4, [1, 2, 3, 0, 4]),
        (None, 4, [1, 2, 3, -4, 4]),
        (-2, 0, [0, 0, 0, -2, 0]),
        (-2, None, [1, 2, 3, -2, 5]),
    ],
)
def test_clip_expr(
    constructor: Constructor, lower: int | None, upper: int | None, expected: list[int]
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(result=nw.col("a").clip(lower_bound=lower, upper_bound=upper))
    assert_equal_data(result, {"result": expected})


def test_clip_expr_expressified(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3, -4, 5], "lb": [3, 2, 1, 1, 1], "ub": [4, 4, 2, 2, 2]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").clip("lb", nw.col("ub") + 1))
    expected_dict = {"a": [3, 2, 3, 1, 3]}
    assert_equal_data(result, expected_dict)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (3, 4, [3, 3, 3, 3, 4]),
        (0, 4, [1, 2, 3, 0, 4]),
        (None, 4, [1, 2, 3, -4, 4]),
        (-2, 0, [0, 0, 0, -2, 0]),
        (-2, None, [1, 2, 3, -2, 5]),
    ],
)
def test_clip_series(
    constructor_eager: ConstructorEager,
    lower: int | None,
    upper: int | None,
    expected: list[int],
) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3, -4, 5]}), eager_only=True)
    result = {"result": df["a"].clip(lower_bound=lower, upper_bound=upper)}

    assert_equal_data(result, {"result": expected})


def test_clip_series_expressified(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3, -4, 5], "lb": [3, 2, 1, 1, 1], "ub": [4, 4, 2, 2, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].clip(df["lb"], df["ub"] + 1).to_frame()
    expected_dict = {"a": [3, 2, 3, 1, 3]}
    assert_equal_data(result, expected_dict)


def test_clip_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises(MultiOutputExpressionError):
        df.select(nw.col("a").clip(nw.all(), nw.col("a", "b")))


def test_clip_preserves_null(constructor: Constructor) -> None:
    # A null value must stay null through clip, not be replaced by a bound.
    df = nw.from_native(constructor({"a": [1, None, 3, -4, None]}))
    result = df.select(a=nw.col("a").clip(lower_bound=0, upper_bound=2))
    assert_equal_data(result, {"a": [1, None, 2, 0, None]})


NULL_BOUNDS_DATA = {
    "a": [1, 3, 10, -4, None],
    "lb": [0, 2, None, 0, 0],
    "ub": [2, None, 3, None, 3],
}
"""Data with nulls inside the bound columns: a null bound means "no bound" for that row."""


def test_clip_expr_null_bounds(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor)
        for x in ["pandas_nullable", "pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas clip cannot fill null bounds for nullable integer dtypes"
            )
        )
    df = nw.from_native(constructor(NULL_BOUNDS_DATA))
    result = df.select(nw.col("a").clip("lb", "ub"))
    assert_equal_data(result, {"a": [1, 3, 3, 0, None]})


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [("lb", None, [1, 3, 10, 0, None]), (None, "ub", [1, 3, 3, -4, None])],
)
def test_clip_expr_null_single_bound(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    lower: str | None,
    upper: str | None,
    expected: list[int | None],
) -> None:
    if any(
        x in str(constructor)
        for x in ["pandas_nullable", "pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas clip cannot fill null bounds for nullable integer dtypes"
            )
        )
    df = nw.from_native(constructor(NULL_BOUNDS_DATA))
    result = df.select(nw.col("a").clip(lower, upper))
    assert_equal_data(result, {"a": expected})


def test_clip_series_null_bounds(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor_eager)
        for x in ["pandas_nullable", "pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas clip cannot fill null bounds for nullable integer dtypes"
            )
        )
    df = nw.from_native(constructor_eager(NULL_BOUNDS_DATA), eager_only=True)
    result = df["a"].clip(df["lb"], df["ub"]).to_frame()
    assert_equal_data(result, {"a": [1, 3, 3, 0, None]})

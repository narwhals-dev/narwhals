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

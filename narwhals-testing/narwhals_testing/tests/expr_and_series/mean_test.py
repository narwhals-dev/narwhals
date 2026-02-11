from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 7], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize("expr", [nw.col("a", "b", "z").mean(), nw.mean("a", "b", "z")])
def test_expr_mean_expr(constructor: Constructor, expr: nw.Expr) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    expected = {"a": [2.0], "b": [5.0], "z": [8.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 2.0), ("b", 5.0), ("z", 8.0)])
def test_expr_mean_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.mean()
    assert_equal_data({col: [result]}, {col: [expected]})

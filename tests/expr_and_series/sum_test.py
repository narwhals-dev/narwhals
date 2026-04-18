from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize("expr", [nw.col("a", "b", "z").sum(), nw.sum("a", "b", "z")])
def test_expr_sum_expr(nw_frame_constructor: Constructor, expr: nw.Expr) -> None:
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(expr)
    expected = {"a": [6], "b": [14], "z": [24.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 6), ("b", 14), ("z", 24.0)])
def test_expr_sum_series(
    nw_eager_constructor: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(nw_eager_constructor(data), eager_only=True)[col]
    result = series.sum()
    assert_equal_data({col: [result]}, {col: [expected]})

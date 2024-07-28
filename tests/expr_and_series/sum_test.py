from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize("expr", [nw.col("a", "b", "z").sum(), nw.sum("a", "b", "z")])
def test_expr_sum_expr(constructor_lazy: Any, expr: nw.Expr) -> None:
    df = nw.from_native(constructor_lazy(data), eager_only=True)
    result = df.select(expr)
    expected = {"a": [6], "b": [14], "z": [24.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 6), ("b", 14), ("z", 24.0)])
def test_expr_sum_series(constructor: Any, col: str, expected: float) -> None:
    series = nw.from_native(constructor(data), eager_only=True)[col]
    result = series.sum()
    compare_dicts({col: [result]}, {col: [expected]})

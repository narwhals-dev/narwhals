from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize("expr", [nw.col("a", "b", "z").max(), nw.max("a", "b", "z")])
def test_expr_max_expr(constructor: Constructor, expr: nw.Expr) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    expected = {"a": [3], "b": [6], "z": [9.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 3), ("b", 6), ("z", 9.0)])
def test_expr_max_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.max()
    compare_dicts({col: [result]}, {col: [expected]})

from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": [3, 8, 2], "b": [5, 5, 7], "z": [7.0, 8, 9]}


@pytest.mark.parametrize(
    "expr", [nw.col("a", "b", "z").median(), nw.median("a", "b", "z")]
)
def test_expr_median_expr(
    constructor: Constructor, expr: nw.Expr, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    expected = {"a": [3.0], "b": [5.0], "z": [8.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 3.0), ("b", 5.0), ("z", 8.0)])
def test_expr_median_series(constructor_eager: Any, col: str, expected: float) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.median()
    compare_dicts({col: [result]}, {col: [expected]})

from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [3, 8, 2, None],
    "b": [5, 5, None, 7],
    "z": [7.0, 8, 9, None],
    "s": ["f", "a", "x", "x"],
}


@pytest.mark.parametrize(
    "expr", [nw.col("a", "b", "z").median(), nw.median("a", "b", "z")]
)
def test_median_expr(
    constructor: Constructor, expr: nw.Expr, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    expected = {"a": [3.0], "b": [5.0], "z": [8.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 3.0), ("b", 5.0), ("z", 8.0)])
def test_median_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.median()
    assert_equal_data({col: [result]}, {col: [expected]})

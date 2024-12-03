from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize(
    "expr", [nw.col("a", "b", "z").argmin(), nw.argmin("a", "b", "z")]
)
def test_expr_argmin_expr(constructor: Constructor, expr: nw.Expr) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    # The index of minimum values: 'a' -> 0 (value 1), 'b' -> 0 (value 4), 'z' -> 0 (value 7.0)
    expected = {"a": [0], "b": [0], "z": [0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 0), ("b", 0), ("z", 0)])
def test_expr_argmin_series(
    constructor_eager: ConstructorEager, col: str, expected: int
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.argmin()
    assert_equal_data({col: [result]}, {col: [expected]})

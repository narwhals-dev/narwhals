from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize(
    "expr", [nw.col("a", "b", "z").argmax(), nw.argmax("a", "b", "z")]
)
def test_expr_argmax_expr(constructor: Constructor, expr: nw.Expr) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    # The index of maximum values: 'a' -> 1 (value 3), 'b' -> 2 (value 6), 'z' -> 2 (value 9)
    expected = {"a": [1], "b": [2], "z": [2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 1), ("b", 2), ("z", 2)])
def test_expr_argmax_series(
    constructor_eager: ConstructorEager, col: str, expected: int
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.argmax()
    assert_equal_data({col: [result]}, {col: [expected]})

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral

data = {"a": [1, 3, None, None], "b": [4, None, 6, None], "z": [3, 1, None, None]}
expected_values = [4, 3, 6, None]


@pytest.mark.filterwarnings(r"ignore:.*All-NaN slice encountered:RuntimeWarning")
def test_maxh(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_max=nw.max_horizontal("a", nw.col("b"), "z"))
    expected = {"horizontal_max": expected_values}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(r"ignore:.*All-NaN slice encountered:RuntimeWarning")
def test_maxh_all(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.max_horizontal(nw.all()), c=nw.max_horizontal(nw.all()))
    expected = {"a": expected_values, "c": expected_values}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("exprs", "name"),
    [
        ((nw.col("a"), 2), "a"),
        ((nw.col("a"), nw.lit(2)), "a"),
        ((2, nw.col("a")), "literal"),
        ((nw.lit(2), nw.col("a")), "literal"),
    ],
)
def test_maxh_with_scalars(
    constructor: Constructor, exprs: tuple[PythonLiteral | nw.Expr, ...], name: str
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.max_horizontal(*exprs))
    assert_equal_data(result, {name: [2, 2, 3]})

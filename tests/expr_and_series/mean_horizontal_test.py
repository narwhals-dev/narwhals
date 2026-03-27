from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import PythonLiteral


def test_meanh(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(nw.col("a"), nw.col("b")))
    expected = {"horizontal_mean": [2.5, 3.0, 6.0, None]}
    assert_equal_data(result, expected)


def test_meanh_with_literal(constructor: Constructor) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(nw.lit(1), "a", nw.col("b")))
    expected = {"horizontal_mean": [2.0, 2.0, 3.5, 1.0]}
    assert_equal_data(result, expected)


def test_meanh_all(constructor: Constructor) -> None:
    data = {"a": [2, 4, 6], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.mean_horizontal(nw.all()))
    expected = {"a": [6, 12, 18]}
    assert_equal_data(result, expected)
    result = df.select(c=nw.mean_horizontal(nw.all()))
    expected = {"c": [6, 12, 18]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("exprs", "name"),
    [
        ((nw.col("a"), 1), "a"),
        ((nw.col("a"), nw.lit(1)), "a"),
        ((1, nw.col("a")), "literal"),
        ((nw.lit(1), nw.col("a")), "literal"),
    ],
)
def test_meanh_with_scalars(
    constructor: Constructor, exprs: tuple[PythonLiteral | nw.Expr, ...], name: str
) -> None:
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.mean_horizontal(*exprs))
    assert_equal_data(result, {name: [1.0, 1.5, 2.0]})

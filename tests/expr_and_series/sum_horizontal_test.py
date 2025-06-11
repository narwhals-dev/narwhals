from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, assert_equal_data


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_sumh(constructor: Constructor, col_expr: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(horizontal_sum=nw.sum_horizontal(col_expr, nw.col("b")))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizontal_sum": [5, 7, 8],
    }
    assert_equal_data(result, expected)


def test_sumh_nullable(constructor: Constructor) -> None:
    data = {"a": [1, 8, 3], "b": [4, 5, None], "i": [0, 1, 2]}
    expected = {"hsum": [5, 13, 3], "i": [0, 1, 2]}

    df = nw.from_native(constructor(data))
    result = df.select(hsum=nw.sum_horizontal("a", "b"), i="i").sort("i")
    assert_equal_data(result, expected)


def test_sumh_all(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.sum_horizontal(nw.all().name.suffix("_foo")))
    expected = {"a_foo": [11, 22, 33]}
    assert_equal_data(result, expected)
    result = df.select(c=nw.sum_horizontal(nw.all()))
    expected = {"c": [11, 22, 33]}
    assert_equal_data(result, expected)


def test_sumh_aggregations(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.sum_horizontal(nw.all().mean().name.suffix("_foo")))
    expected = {"a_foo": [22]}
    assert_equal_data(result, expected)


def test_sumh_transformations(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.select(d=nw.sum_horizontal("a", nw.col("b").sum(), "c"))
    expected: dict[str, Any] = {"d": [23, 25, 27]}
    assert_equal_data(result, expected)
    result = df.select(d=nw.sum_horizontal("a", nw.lit(None, dtype=nw.Float64), "c"))
    expected = {"d": [8.0, 10.0, 12.0]}
    assert_equal_data(result, expected)

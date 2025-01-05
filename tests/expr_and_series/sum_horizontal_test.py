from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_sumh(
    constructor: Constructor, col_expr: Any, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(horizontal_sum=nw.sum_horizontal(col_expr, nw.col("b")))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizontal_sum": [5, 7, 8],
    }
    assert_equal_data(result, expected)


def test_sumh_nullable(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 8, 3], "b": [4, 5, None]}
    expected = {"hsum": [5, 13, 3]}

    df = nw.from_native(constructor(data))
    result = df.select(hsum=nw.sum_horizontal("a", "b"))
    assert_equal_data(result, expected)


def test_sumh_all(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.sum_horizontal(nw.all()))
    expected = {
        "a": [11, 22, 33],
    }
    assert_equal_data(result, expected)
    result = df.select(c=nw.sum_horizontal(nw.all()))
    expected = {
        "c": [11, 22, 33],
    }
    assert_equal_data(result, expected)

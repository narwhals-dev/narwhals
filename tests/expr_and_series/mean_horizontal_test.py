from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_meanh(constructor: Constructor, col_expr: Any) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(col_expr, nw.col("b")))
    expected = {"horizontal_mean": [2.5, 3.0, 6.0, None]}
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

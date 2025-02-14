from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 3, None, None], "b": [4, None, 6, None], "z": [3, 1, None, None]}
expected_values = [1, 1, 6, None]


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
@pytest.mark.filterwarnings(r"ignore:.*All-NaN slice encountered:RuntimeWarning")
def test_minh(constructor: Constructor, col_expr: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_min=nw.min_horizontal(col_expr, nw.col("b"), "z"))
    expected = {"horizontal_min": expected_values}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(r"ignore:.*All-NaN slice encountered:RuntimeWarning")
def test_minh_all(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.min_horizontal(nw.all()), c=nw.min_horizontal(nw.all()))
    expected = {"a": expected_values, "c": expected_values}
    assert_equal_data(result, expected)

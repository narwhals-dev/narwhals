from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": [1, 3, None, None], "b": [4, None, 6, None], "z": [3, 1, None, None]}
expcted_values = [1, 1, 6, float("nan")]


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_minh(constructor: Constructor, col_expr: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_min=nw.min_horizontal(col_expr, nw.col("b"), "z"))
    expected = {"horizontal_min": expcted_values}
    compare_dicts(result, expected)


def test_minh_all(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.min_horizontal(nw.all()), c=nw.min_horizontal(nw.all()))
    expected = {"a": expcted_values, "c": expcted_values}
    compare_dicts(result, expected)

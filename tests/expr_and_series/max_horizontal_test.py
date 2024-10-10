from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": [1, 3, None, None], "b": [4, None, 6, None], "z": [3, 1, None, None]}
expected_values = [4, 3, 6, float("nan")]


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_maxh(constructor: Constructor, col_expr: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_max=nw.max_horizontal(col_expr, nw.col("b"), "z"))
    expected = {"horizontal_max": expected_values}
    compare_dicts(result, expected)


def test_maxh_all(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.max_horizontal(nw.all()), c=nw.max_horizontal(nw.all()))
    expected = {"a": expected_values, "c": expected_values}
    compare_dicts(result, expected)

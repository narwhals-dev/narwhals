from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_meanh(constructor: Any, col_expr: Any) -> None:
    data = {"a": [1, 3, None, None], "b": [4, None, 6, None]}
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_mean=nw.mean_horizontal(col_expr, nw.col("b")))
    expected = {"horizontal_mean": [2.5, 3.0, 6.0, float("nan")]}
    compare_dicts(result, expected)

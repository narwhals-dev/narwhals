from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_sumh(constructor_lazy: Any, col_expr: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_lazy(data))
    result = df.with_columns(horizontal_sum=nw.sum_horizontal(col_expr, nw.col("b")))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizontal_sum": [5, 7, 8],
    }
    compare_dicts(result, expected)

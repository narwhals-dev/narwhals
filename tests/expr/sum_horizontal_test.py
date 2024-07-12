from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize("col_expr", [nw.col("a"), "a"])
def test_sumh(constructor_with_pyarrow: Any, col_expr: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df.with_columns(horizonal_sum=nw.sum_horizontal(col_expr, nw.col("b")))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizonal_sum": [5, 7, 8],
    }
    compare_dicts(result, expected)

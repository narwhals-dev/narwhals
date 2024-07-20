from __future__ import annotations
from tests.utils import compare_dicts

from typing import Any

import narwhals as nw

def test_drop_nulls(constructor: Any) -> None:
    data = {
        'A': [1, 2, None, 4],
        'B': [5, 6, 7, 8],
        'C': [None, None, None, None],
        'D': [9, 10, 11, 12]
    }

    df = nw.from_native(
        constructor(data), strict=False, eager_only=True, allow_series=True
    )

    result_a = df.select(nw.col("A").drop_nulls())
    result_b = df.select(nw.col("B").drop_nulls())
    result_c = df.select(nw.col("C").drop_nulls())
    result_d = df.select(nw.col("D").drop_nulls())
    expected_a = {"A": [1.0, 2.0, 4.0]}
    expected_b = {"B": [5, 6, 7, 8]}
    expected_c = {"C": []}
    expected_d = {"D": [9, 10, 11, 12]}

    compare_dicts(result_a, expected_a)
    compare_dicts(result_b, expected_b)
    compare_dicts(result_c, expected_c)
    compare_dicts(result_d, expected_d)
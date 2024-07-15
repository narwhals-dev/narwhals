from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_series_sum(constructor_with_pyarrow: Any) -> None:
    data = {
        "a": [0, 1, 2, 3, 4],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, None, 2, 1],
    }
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True, allow_series=True)

    # Calculating the sum for each column seperately
    result_sum_a = df.select(nw.col("a").sum())
    result_sum_b = df.select(nw.col("b").sum())
    result_sum_c = df.select(nw.col("c").sum())

    # Convert the results to Narwhals Native Frame
    result_native_a = nw.to_native(result_sum_a)
    result_native_b = nw.to_native(result_sum_b)
    result_native_c = nw.to_native(result_sum_c)

    expected_sum_a = {"a": [10]}
    expected_sum_b = {"b": [14]}
    expected_sum_c = {"c": [12]}

    compare_dicts(result_native_a, expected_sum_a)
    compare_dicts(result_native_b, expected_sum_b)
    compare_dicts(result_native_c, expected_sum_c)

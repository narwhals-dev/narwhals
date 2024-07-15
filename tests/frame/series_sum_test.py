from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_series_sum(constructor: Any) -> None:
    data = {
        "a": [0, 1, 2, 3, 4],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, None, 2, 1],
    }
    df = nw.from_native(constructor(data), eager_only=True, allow_series=True)

    # Calculating the sum for each column seperately
    result = df.select(nw.col("a", "b", "c").sum())

    expected_sum = {"a": [10], "b": [14], "c": [12]}

    compare_dicts(result, expected_sum)

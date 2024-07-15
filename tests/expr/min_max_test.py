from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_expr_min_max(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data), eager_only=True)
    result_min = df.select(nw.col("a", "b", "z").min())
    result_max = df.select(nw.col("a", "b", "z").max())
    expected_min = {"a": [1], "b": [4], "z": [7.0]}
    expected_max = {"a": [3], "b": [6], "z": [9]}
    compare_dicts(result_min, expected_min)
    compare_dicts(result_max, expected_max)

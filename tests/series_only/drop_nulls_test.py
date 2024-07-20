from __future__ import annotations

from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


def test_drop_nulls(constructor: Any) -> None:
    data = {
        "A": [1, 2, None, 4],
        "B": [5, 6, 7, 8],
        "C": [None, None, None, None],
        "D": [9, 10, 11, 12],
    }

    df = nw.from_native(
        constructor(data), strict=False, eager_only=True, allow_series=True
    )

    result = df.select(nw.col("A", "B", "C", "D").drop_nulls())
    expected = {
        "A": [1.0, 2.0, 4.0],
        "B": [5, 6, 8],
        "C": [],
        "D": [9, 10, 12],
    }

 
    compare_dicts(result, expected)

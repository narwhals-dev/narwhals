from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_null_count(constructor: Any) -> None:
    data = {"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = df.null_count()
    expected = {"a": [1], "b": [0], "z": [1]}
    compare_dicts(result, expected)

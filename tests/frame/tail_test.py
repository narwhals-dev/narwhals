from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_tail(request: pytest.FixtureRequest, constructor: Any) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9]}

    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()

    result = df.tail(2)
    compare_dicts(result, expected)

    result = df.collect().tail(2)  # type: ignore[assignment]
    compare_dicts(result, expected)

    result = df.collect().tail(-1)  # type: ignore[assignment]
    compare_dicts(result, expected)

    result = df.collect().select(nw.col("a").tail(2))  # type: ignore[assignment]
    compare_dicts(result, {"a": expected["a"]})

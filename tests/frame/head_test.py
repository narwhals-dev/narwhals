from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_head(constructor_eager: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    expected = {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}

    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw).lazy()

    result = df.head(2)
    compare_dicts(result, expected)

    result = df.collect().head(2)  # type: ignore[assignment]
    compare_dicts(result, expected)

    result = df.collect().head(-1)  # type: ignore[assignment]
    compare_dicts(result, expected)

    result = df.collect().select(nw.col("a").head(2))  # type: ignore[assignment]
    compare_dicts(result, {"a": expected["a"]})

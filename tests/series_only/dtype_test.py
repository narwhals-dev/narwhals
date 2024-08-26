from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw

data = {"a": [1, 3, 2]}


def test_dtype(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.dtype
    assert result == nw.Int64
    assert result.is_numeric()

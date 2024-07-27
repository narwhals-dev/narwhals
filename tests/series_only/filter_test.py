from __future__ import annotations

from typing import Any

import numpy as np

import narwhals as nw


def test_filter(constructor: Any) -> None:
    data = [1, 3, 2]
    series = nw.from_native(constructor({"a": data}), eager_only=True)["a"]
    result = series.filter(series > 1)
    expected = np.array([3, 2])
    assert (result.to_numpy() == expected).all()

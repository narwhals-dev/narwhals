from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

import narwhals as nw
from narwhals.dependencies import is_into_series

if TYPE_CHECKING:
    from typing_extensions import Self


class ListBackedSeries:
    def __init__(self, name: str, data: list[Any]) -> None:
        self._data = data
        self._name = name

    def __len__(self) -> int:
        return len(self._data)

    def __narwhals_series__(self) -> Self:
        return self


def test_is_into_series() -> None:
    assert is_into_series(pa.chunked_array([["a", "b"]]))
    assert is_into_series(pl.Series([1, 2, 3]))
    assert is_into_series(pd.Series([1, 2, 3]))
    assert is_into_series(nw.from_native(pd.Series([1, 2, 3]), series_only=True))
    assert is_into_series(ListBackedSeries("a", [1, 4, 2]))
    assert not is_into_series(np.array([1, 2, 3]))
    assert not is_into_series([1, 2, 3])

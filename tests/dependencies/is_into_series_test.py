from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import narwhals as nw
from narwhals.stable.v1.dependencies import is_into_series

if TYPE_CHECKING:
    from typing_extensions import Self


class ListBackedSeries:
    def __init__(self, name: str, data: list[Any]) -> None:
        self._data = data
        self._name = name

    def __len__(self) -> int:  # pragma: no cover
        return len(self._data)

    def __narwhals_series__(self) -> Self:  # pragma: no cover
        return self


def test_is_into_series_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    assert is_into_series(pa.chunked_array([["a", "b"]]))


def test_is_into_series_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    assert is_into_series(pl.Series([1, 2, 3]))


def test_is_into_series_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    assert is_into_series(pd.Series([1, 2, 3]))
    assert is_into_series(nw.from_native(pd.Series([1, 2, 3]), series_only=True))


def test_is_into_series() -> None:
    assert is_into_series(ListBackedSeries("a", [1, 4, 2]))
    assert not is_into_series(np.array([1, 2, 3]))
    assert not is_into_series([1, 2, 3])

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import narwhals as nw
from narwhals.stable.v1.dependencies import is_into_dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

DATA: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class DictDataFrame:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __len__(self) -> int:  # pragma: no cover
        return len(next(iter(self._data.values())))

    def __narwhals_dataframe__(self) -> Self:  # pragma: no cover
        return self


def test_is_into_dataframe_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    assert is_into_dataframe(pa.table(DATA))


def test_is_into_dataframe_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    assert is_into_dataframe(pl.DataFrame(DATA))


def test_is_into_dataframe_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    assert is_into_dataframe(pd.DataFrame(DATA))
    assert is_into_dataframe(nw.from_native(pd.DataFrame(DATA)))


def test_is_into_dataframe_other() -> None:
    assert is_into_dataframe(DictDataFrame(DATA))
    assert not is_into_dataframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not is_into_dataframe(DATA)

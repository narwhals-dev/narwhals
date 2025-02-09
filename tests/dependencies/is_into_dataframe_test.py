from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

import narwhals as nw
from narwhals.stable.v1.dependencies import is_into_dataframe

if TYPE_CHECKING:
    from typing_extensions import Self


class DictDataFrame:
    def __init__(self: Self, data: dict[str, list[Any]]) -> None:
        self._data = data

    def __len__(self) -> int:  # pragma: no cover
        return len(next(iter(self._data.values())))

    def __narwhals_dataframe__(self) -> Self:  # pragma: no cover
        return self


def test_is_into_dataframe() -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert is_into_dataframe(pa.table(data))
    assert is_into_dataframe(pl.DataFrame(data))
    assert is_into_dataframe(pd.DataFrame(data))
    assert is_into_dataframe(nw.from_native(pd.DataFrame(data)))
    assert is_into_dataframe(DictDataFrame(data))
    assert not is_into_dataframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not is_into_dataframe(data)

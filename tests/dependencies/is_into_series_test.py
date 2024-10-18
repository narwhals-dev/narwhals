from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

import narwhals as nw
from narwhals.dependencies import is_into_series


def test_is_into_series() -> None:
    assert is_into_series(pa.chunked_array([["a", "b"]]))
    assert is_into_series(pl.Series([1, 2, 3]))
    assert is_into_series(pd.Series([1, 2, 3]))
    assert is_into_series(nw.from_native(pd.Series([1, 2, 3]), allow_series=True))
    assert not is_into_series(np.array([1, 2, 3]))
    assert not is_into_series([1, 2, 3])

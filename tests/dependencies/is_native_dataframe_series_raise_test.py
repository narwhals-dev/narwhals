from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import pytest

import narwhals as nw
from narwhals.dependencies import (
    is_cudf_dataframe,
    is_cudf_series,
    is_ibis_table,
    is_modin_dataframe,
    is_modin_series,
    is_pandas_dataframe,
    is_pandas_like_dataframe,
    is_pandas_like_series,
    is_pandas_series,
    is_polars_dataframe,
    is_polars_lazyframe,
    is_polars_series,
    is_pyarrow_chunked_array,
    is_pyarrow_table,
)


@pytest.mark.parametrize(
    "is_native_dataframe",
    [
        is_pandas_dataframe,
        is_modin_dataframe,
        is_polars_dataframe,
        is_cudf_dataframe,
        is_ibis_table,
        is_polars_lazyframe,
        is_pyarrow_table,
        is_pandas_like_dataframe,
    ],
)
def test_is_native_dataframe(is_native_dataframe: Callable[[Any], Any]) -> None:
    data = {"a": [1, 2], "b": ["bar", "foo"]}
    df = nw.from_native(pd.DataFrame(data))
    with pytest.raises(TypeError, match="did you mean"):
        is_native_dataframe(df)


@pytest.mark.parametrize(
    "is_native_series",
    [
        is_pandas_series,
        is_modin_series,
        is_polars_series,
        is_cudf_series,
        is_pyarrow_chunked_array,
        is_pandas_like_series,
    ],
)
def test_is_native_series(is_native_series: Callable[[Any], Any]) -> None:
    data = {"a": [1, 2]}
    ser = nw.from_native(pd.DataFrame(data))["a"]
    with pytest.raises(TypeError, match="did you mean"):
        is_native_series(ser)

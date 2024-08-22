# pandas / Polars / etc. : if a user passes a dataframe from one of these
# libraries, it means they must already have imported the given module.
# So, we can just check sys.modules.
from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import numpy as np

    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
    import cudf
    import dask.dataframe as dd
    import modin.pandas as mpd
    import pandas as pd
    import polars as pl
    import pyarrow as pa


def get_polars() -> Any:
    """Get Polars module (if already imported - else return None)."""
    return sys.modules.get("polars", None)


def get_pandas() -> Any:
    """Get pandas module (if already imported - else return None)."""
    return sys.modules.get("pandas", None)


def get_modin() -> Any:  # pragma: no cover
    """Get modin.pandas module (if already imported - else return None)."""
    if (modin := sys.modules.get("modin", None)) is not None:
        return modin.pandas
    return None


def get_cudf() -> Any:
    """Get cudf module (if already imported - else return None)."""
    return sys.modules.get("cudf", None)


def get_pyarrow() -> Any:  # pragma: no cover
    """Get pyarrow module (if already imported - else return None)."""
    return sys.modules.get("pyarrow", None)


def get_numpy() -> Any:
    """Get numpy module (if already imported - else return None)."""
    return sys.modules.get("numpy", None)


def get_dask() -> Any:
    """Get dask (if already imported - else return None)."""
    return sys.modules.get("dask", None)


def get_dask_dataframe() -> Any:
    """Get dask.dataframe module (if already imported - else return None)."""
    return sys.modules.get("dask.dataframe", None)


def get_dask_expr() -> Any:
    """Get dask_expr module (if already imported - else return None)."""
    return sys.modules.get("dask_expr", None)


def is_pandas_dataframe(df: Any) -> TypeGuard[pd.DataFrame]:
    """Check whether `df` is a pandas DataFrame without importing pandas."""
    return bool((pd := get_pandas()) is not None and isinstance(df, pd.DataFrame))


def is_pandas_series(ser: Any) -> TypeGuard[pd.Series[Any]]:
    """Check whether `ser` is a pandas Series without importing pandas."""
    return bool((pd := get_pandas()) is not None and isinstance(ser, pd.Series))


def is_modin_dataframe(df: Any) -> TypeGuard[mpd.DataFrame]:
    """Check whether `df` is a modin DataFrame without importing modin."""
    return bool((pd := get_modin()) is not None and isinstance(df, pd.DataFrame))


def is_modin_series(ser: Any) -> TypeGuard[mpd.Series]:
    """Check whether `ser` is a modin Series without importing modin."""
    return bool((pd := get_modin()) is not None and isinstance(ser, pd.Series))


def is_cudf_dataframe(df: Any) -> TypeGuard[cudf.DataFrame]:
    """Check whether `df` is a cudf DataFrame without importing cudf."""
    return bool((pd := get_cudf()) is not None and isinstance(df, pd.DataFrame))


def is_cudf_series(ser: Any) -> TypeGuard[pd.Series[Any]]:
    """Check whether `ser` is a cudf Series without importing cudf."""
    return bool((pd := get_cudf()) is not None and isinstance(ser, pd.Series))


def is_dask_dataframe(df: Any) -> TypeGuard[dd.DataFrame]:
    """Check whether `df` is a Dask DataFrame without importing Dask."""
    return bool((dd := get_dask_dataframe()) is not None and isinstance(df, dd.DataFrame))


def is_polars_dataframe(df: Any) -> TypeGuard[pl.DataFrame]:
    """Check whether `df` is a Polars DataFrame without importing Polars."""
    return bool((pl := get_polars()) is not None and isinstance(df, pl.DataFrame))


def is_polars_lazyframe(df: Any) -> TypeGuard[pl.LazyFrame]:
    """Check whether `df` is a Polars LazyFrame without importing Polars."""
    return bool((pl := get_polars()) is not None and isinstance(df, pl.LazyFrame))


def is_polars_series(ser: Any) -> TypeGuard[pl.Series]:
    """Check whether `ser` is a Polars Series without importing Polars."""
    return bool((pl := get_polars()) is not None and isinstance(ser, pl.Series))


def is_pyarrow_chunked_array(ser: Any) -> TypeGuard[pa.ChunkedArray]:
    """Check whether `ser` is a PyArrow ChunkedArray without importing PyArrow."""
    return bool((pa := get_pyarrow()) is not None and isinstance(ser, pa.ChunkedArray))


def is_pyarrow_table(df: Any) -> TypeGuard[pa.Table]:
    """Check whether `df` is a PyArrow Table without importing PyArrow."""
    return bool((pa := get_pyarrow()) is not None and isinstance(df, pa.Table))


def is_numpy_array(arr: Any) -> TypeGuard[np.ndarray]:
    """Check whether `arr` is a NumPy Array without importing NumPy."""
    return bool((np := get_numpy()) is not None and isinstance(arr, np.ndarray))


def is_pandas_like_dataframe(df: Any) -> bool:
    """
    Check whether `df` is a pandas-like DataFrame without doing any imports

    By "pandas-like", we mean: pandas, Modin, cuDF.
    """
    return is_pandas_dataframe(df) or is_modin_dataframe(df) or is_cudf_dataframe(df)


def is_pandas_like_series(arr: Any) -> bool:
    """
    Check whether `arr` is a pandas-like Series without doing any imports

    By "pandas-like", we mean: pandas, Modin, cuDF.
    """
    return is_pandas_series(arr) or is_modin_series(arr) or is_cudf_series(arr)


__all__ = [
    "get_polars",
    "get_pandas",
    "get_modin",
    "get_cudf",
    "get_pyarrow",
    "get_numpy",
    "is_pandas_dataframe",
    "is_pandas_series",
    "is_polars_dataframe",
    "is_polars_lazyframe",
    "is_polars_series",
    "is_modin_dataframe",
    "is_modin_series",
    "is_cudf_dataframe",
    "is_cudf_series",
    "is_pyarrow_table",
    "is_pyarrow_chunked_array",
    "is_numpy_array",
    "is_dask_dataframe",
    "is_pandas_like_dataframe",
    "is_pandas_like_series",
]

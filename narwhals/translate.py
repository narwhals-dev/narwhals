from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from narwhals.dataframe import BaseFrame
    from narwhals.series import Series


def to_native(obj: BaseFrame | Series) -> Any:
    from narwhals.dataframe import BaseFrame
    from narwhals.series import Series

    if isinstance(obj, BaseFrame):
        return (
            obj._dataframe
            if obj._implementation == "polars"
            else obj._dataframe._dataframe
        )
    if isinstance(obj, Series):
        return obj._series if obj._implementation == "polars" else obj._series._series

    msg = f"Expected Narwhals object, got {type(obj)}."
    raise TypeError(msg)


def from_native(df: Any) -> BaseFrame:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame

    if (pl := get_polars()) is not None and isinstance(df, pl.DataFrame):
        return DataFrame(df)
    elif (pl := get_polars()) is not None and isinstance(df, pl.LazyFrame):
        return LazyFrame(df)
    elif (
        (pd := get_pandas()) is not None
        and isinstance(df, pd.DataFrame)
        or (mpd := get_modin()) is not None
        and isinstance(df, mpd.DataFrame)
        or (cudf := get_cudf()) is not None
        and isinstance(df, cudf.DataFrame)
    ):
        return DataFrame(df)
    elif hasattr(df, "__narwhals_dataframe__"):  # pragma: no cover
        return DataFrame(df.__narwhals_dataframe__())
    elif hasattr(df, "__narwhals_lazyframe__"):  # pragma: no cover
        return LazyFrame(df.__narwhals_lazyframe__())
    else:
        msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(df)}"
        raise TypeError(msg)


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "get_cudf",
    "to_native",
]

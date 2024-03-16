from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.series import Series
    from narwhals.typing import T


def to_native(obj: DataFrame[T] | Series[T]) -> T:
    from narwhals.dataframe import DataFrame
    from narwhals.series import Series

    if isinstance(obj, DataFrame):
        return (  # type: ignore[no-any-return]
            obj._dataframe
            if obj._implementation == "polars"
            else obj._dataframe._dataframe
        )
    if isinstance(obj, Series):
        return obj._series if obj._implementation == "polars" else obj._series._series  # type: ignore[no-any-return]

    msg = f"Expected Narwhals object, got {type(obj)}."
    raise TypeError(msg)


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "to_native",
]

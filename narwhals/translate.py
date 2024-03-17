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


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "get_cudf",
    "to_native",
]

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from narwhals._interchange.dataframe import map_interchange_dtype_to_narwhals_dtype

if TYPE_CHECKING:
    from narwhals import dtypes


class InterchangeSeries:
    def __init__(self, df: Any) -> None:
        self._native_series = df

    def __narwhals_series__(self) -> Any:
        return self

    @property
    def dtype(self) -> dtypes.DType:
        return map_interchange_dtype_to_narwhals_dtype(self._native_series.dtype)

    def __getattr__(self, attr: str) -> NoReturn:
        msg = (
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "Hint: you probably called `nw.from_native` on an object which isn't fully "
            "supported by Narwhals, yet implements `__dataframe__`. If you would like to "
            "see this kind of object supported in Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)

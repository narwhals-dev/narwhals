from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._ibis.dataframe import map_ibis_dtype_to_narwhals_dtype

if TYPE_CHECKING:
    from narwhals.dtypes import DType


class IbisInterchangeSeries:
    def __init__(self, df: Any) -> None:
        self._native_series = df

    def __narwhals_series__(self) -> Any:
        return self

    @property
    def dtype(self) -> DType:
        return map_ibis_dtype_to_narwhals_dtype(self._native_series.type(), self._dtypes)

    def __getattr__(self, attr: str) -> Any:
        msg = (
            f"Attribute {attr} is not supported for metadata-only dataframes.\n\n"
            "If you would like to see this kind of object better supported in "
            "Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)

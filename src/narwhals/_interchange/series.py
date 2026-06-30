from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NoReturn

from narwhals._interchange.dataframe import (
    map_interchange_dtype_to_narwhals_dtype,
    unsupported_error,
)
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType


class InterchangeSeries:
    _version = Version.V1
    _implementation: Final = Implementation.UNKNOWN

    def __init__(self, df: Any) -> None:
        self._native_series = df

    def __narwhals_series__(self) -> Self:
        return self

    def __native_namespace__(self) -> NoReturn:
        msg = "Cannot access native namespace for interchange-level series with unknown backend."
        raise NotImplementedError(msg)

    @property
    def dtype(self) -> DType:
        return map_interchange_dtype_to_narwhals_dtype(self._native_series.dtype)

    @property
    def native(self) -> Any:
        return self._native_series

    def __getattr__(self, attr: str) -> NoReturn:
        raise unsupported_error(attr)  # pragma: no cover

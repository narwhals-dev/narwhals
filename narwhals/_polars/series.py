from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import reverse_translate_dtype

PL = get_polars()


class PolarsSeries:
    def __init__(self, series: Any) -> None:
        self._native_series = series

    def __repr__(self) -> str:
        return "PolarsSeries"

    def __narwhals_series__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace()

    def _from_native_series(self, series: Any) -> Self:
        return self.__class__(series)

    def _from_native_object(self, series: Any) -> Any:
        pl = get_polars()
        if isinstance(series, pl.Series):
            return self._from_native_series(series)
        if isinstance(series, pl.DataFrame):
            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(series)
        if isinstance(series, pl.LazyFrame):
            from narwhals._polars.dataframe import PolarsLazyFrame

            return PolarsLazyFrame(series)
        # scalar
        return series

    def __getattr__(self, attr: str) -> Any:
        if attr == "as_py":
            raise AttributeError
        return lambda *args, **kwargs: self._from_native_object(
            getattr(self._native_series, attr)(*args, **kwargs)
        )

    def __len__(self) -> int:
        return len(self._native_series)

    def __getitem__(self, item: Any) -> Any:
        return self._from_native_object(self._native_series.__getitem__(item))

    def cast(self, dtype: DType) -> Self:
        ser = self._native_series
        dtype = reverse_translate_dtype(dtype)
        return self._from_native_series(ser.cast(dtype))

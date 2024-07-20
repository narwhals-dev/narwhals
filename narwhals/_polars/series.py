from __future__ import annotations
from narwhals.dependencies import get_polars

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.dtypes import DType

from narwhals._polars.utils import reverse_translate_dtype 
from narwhals._polars.namespace import PolarsNamespace

PL = get_polars()


class PolarsSeries:
    def __init__(self, series):
        self._native_series = series

    def __repr__(self):
        return "PolarsSeries"

    def __narwhals_series__(self):
        return self

    def __narwhals_namespace__(self):
        return PolarsNamespace()

    def _from_native_series(self, series):
        return self.__class__(series)

    def _from_native_object(self, series):
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

    def __getattr__(self, attr):
        if attr == "as_py":
            raise AttributeError
        return lambda *args, **kwargs: self._from_native_object(
            getattr(self._native_series, attr)(*args, **kwargs)
        )

    def __len__(self):
        return len(self._native_series)

    def __getitem__(self, item):
        return self._from_native_object(self._native_series.__getitem__(item))

    def cast(self, dtype: DType) -> Self:
        ser = self._native_series
        dtype = reverse_translate_dtype(dtype)
        return self._from_native_series(ser.cast(dtype))

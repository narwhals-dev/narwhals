from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dtypes import to_narwhals_dtype
from narwhals.dtypes import translate_dtype
from narwhals.translate import get_pandas
from narwhals.translate import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self


class Series:
    def __init__(
        self,
        series: Any,
        *,
        is_polars: bool = False,
    ) -> None:
        from narwhals._pandas_like.series import PandasSeries

        self._is_polars = is_polars
        if hasattr(series, "__narwhals_series__"):
            self._series = series.__narwhals_series__()
            return
        if is_polars or (
            (pl := get_polars()) is not None and isinstance(series, pl.Series)
        ):
            self._series = series
            self._is_polars = True
            return
        if (pd := get_pandas()) is not None and isinstance(series, pd.Series):
            self._series = PandasSeries(series, implementation="pandas")
            return
        msg = f"Expected pandas or Polars Series, got: {type(series)}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    def __narwhals_namespace__(self) -> Any:
        if self._is_polars:
            import polars as pl

            return pl
        return self._series.__narwhals_namespace__()

    def _extract_native(self, arg: Any) -> Any:
        from narwhals.series import Series

        if isinstance(arg, Series):
            return arg._series
        return arg

    def _from_series(self, series: Any) -> Self:
        return self.__class__(series, is_polars=self._is_polars)

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals Series                                 "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `narwhals.to_native()` to see native output |\n"
            + "└"
            + "─" * length
            + "┘"
        )

    def __len__(self) -> int:
        return len(self._series)

    @property
    def dtype(self) -> Any:
        return to_narwhals_dtype(self._series.dtype, is_polars=self._is_polars)

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        return self._from_series(
            self._series.cast(translate_dtype(self.__narwhals_namespace__(), dtype))
        )

    def mean(self) -> Any:
        return self._series.mean()

    def std(self) -> Any:
        return self._series.std()

    def is_in(self, other: Any) -> Self:
        return self._from_series(self._series.is_in(self._extract_native(other)))

    def sort(self) -> Self:
        return self._from_series(self._series.sort())

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        return self._series.to_pandas()

    def __gt__(self, other: Any) -> Series:
        return self._from_series(self._series.__gt__(self._extract_native(other)))

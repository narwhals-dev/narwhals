from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

import cudf as pd
from pandas.api.types import is_extension_array_dtype

import polars_api_compat
from polars_api_compat.spec import Series as SeriesT
from polars_api_compat.utils import validate_column_comparand

if TYPE_CHECKING:
    import cudf as pd


class Series(SeriesT):
    def __init__(
        self,
        series: pd.Series[Any],
        *,
        api_version: str,
    ) -> None:
        """Parameters
        ----------
        df
            DataFrame this column originates from.
        """

        self._name = series.name
        assert self._name is not None
        self._series = series
        self.api_version = api_version

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard Column (api_version={self.api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.column` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def __iter__(self) -> NoReturn:
        msg = ""
        raise NotImplementedError(msg)

    def _from_series(self, series: pd.Series) -> Series:
        return Series(
            series.rename(series.name, copy=False),
            api_version=self.api_version,
        )

    # In the standard
    def __series_namespace__(
        self,
    ) -> polars_api_compat.cudf.Namespace:
        return polars_api_compat.cudf.Namespace(
            api_version=self.api_version,
        )

    @property
    def name(self) -> str:
        return self._name  # type: ignore[no-any-return]

    @property
    def series(self) -> pd.Series[Any]:
        return self._series

    def filter(self, mask: Series) -> Series:
        ser = self.series
        return self._from_series(ser.loc[validate_column_comparand(self, mask)])

    def item(self) -> Any:
        assert len(self.series) == 1
        return self.series.iloc[0]

    # Binary comparisons

    def __eq__(self, other: object) -> Series:  # type: ignore[override]
        other = validate_column_comparand(self, other)
        ser = self.series
        return self._from_series((ser == other).rename(ser.name, copy=False))

    def __ne__(self, other: object) -> Series:  # type: ignore[override]
        other = validate_column_comparand(self, other)
        ser = self.series
        return self._from_series((ser != other).rename(ser.name, copy=False))

    def __ge__(self, other: Any) -> Series:
        other = validate_column_comparand(self, other)
        ser = self.series
        return self._from_series((ser >= other).rename(ser.name, copy=False))

    def __gt__(self, other: Any) -> Series:
        other = validate_column_comparand(self, other)
        ser = self.series
        return self._from_series((ser > other).rename(ser.name, copy=False))

    def __le__(self, other: Any) -> Series:
        other = validate_column_comparand(self, other)
        ser = self.series
        return self._from_series((ser <= other).rename(ser.name, copy=False))

    def __lt__(self, other: Any) -> Series:
        other = validate_column_comparand(self, other)
        ser = self.series
        return self._from_series((ser < other).rename(ser.name, copy=False))

    def __and__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser & other).rename(ser.name, copy=False))

    def __rand__(self, other: Any) -> Series:
        return self.__and__(other)

    def __or__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser | other).rename(ser.name, copy=False))

    def __ror__(self, other: Any) -> Series:
        return self.__or__(other)

    def __add__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser + other).rename(ser.name, copy=False))

    def __radd__(self, other: Any) -> Series:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser - other).rename(ser.name, copy=False))

    def __rsub__(self, other: Any) -> Series:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser * other).rename(ser.name, copy=False))

    def __rmul__(self, other: Any) -> Series:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser / other).rename(ser.name, copy=False))

    def __rtruediv__(self, other: Any) -> Series:
        raise NotImplementedError

    def __floordiv__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser // other).rename(ser.name, copy=False))

    def __rfloordiv__(self, other: Any) -> Series:
        raise NotImplementedError

    def __pow__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser**other).rename(ser.name, copy=False))

    def __rpow__(self, other: Any) -> Series:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(self, other)
        return self._from_series((ser % other).rename(ser.name, copy=False))

    def __rmod__(self, other: Any) -> Series:  # pragma: no cover
        raise NotImplementedError

    # Unary

    def __invert__(self: Series) -> Series:
        ser = self.series
        return self._from_series(~ser)

    # Reductions

    def any(self) -> Any:
        ser = self.series
        return ser.any()

    def all(self) -> Any:
        ser = self.series
        return ser.all()

    def min(self) -> Any:
        ser = self.series
        return ser.min()

    def max(self) -> Any:
        ser = self.series
        return ser.max()

    def sum(self) -> Any:
        ser = self.series
        return ser.sum()

    def prod(self) -> Any:
        ser = self.series
        return ser.prod()

    def median(self) -> Any:
        ser = self.series
        return ser.median()

    def mean(self) -> Any:
        ser = self.series
        return ser.mean()

    def std(
        self,
        *,
        correction: float = 1.0,
    ) -> Any:
        ser = self.series
        return ser.std(ddof=correction)

    def var(
        self,
        *,
        correction: float = 1.0,
    ) -> Any:
        ser = self.series
        return ser.var(ddof=correction)

    def len(self) -> Any:
        return len(self._series)

    # Transformations

    def is_null(self) -> Series:
        ser = self.series
        return self._from_series(ser.isna())

    def is_nan(self) -> Series:
        ser = self.series
        if is_extension_array_dtype(ser.dtype):
            return self._from_series((ser != ser).fillna(False))  # noqa: PLR0124
        return self._from_series(ser.isna())

    def sort(
        self,
        *,
        descending: bool = True,
    ) -> Series:
        ser = self.series
        return self._from_series(
            ser.sort_values(ascending=not descending).rename(self.name)
        )

    def alias(self, name: str) -> Series:
        ser = self.series
        return self._from_series(ser.rename(name, copy=False))

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from pandas.api.types import is_extension_array_dtype

from narwhals.pandas_like.utils import item
from narwhals.pandas_like.utils import reverse_translate_dtype
from narwhals.pandas_like.utils import validate_column_comparand
from narwhals.spec import Series as SeriesProtocol

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.pandas_like.dtypes import DType


class Series(SeriesProtocol):
    def __init__(
        self,
        series: Any,
        *,
        implementation: str,
    ) -> None:
        """Parameters
        ----------
        df
            DataFrame this column originates from.
        """

        self._name = series.name
        assert self._name is not None
        self._series = series.reset_index(drop=True)
        self._implementation = implementation

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals Series                         "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.to_native()` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def _from_series(self, series: Any) -> Self:
        return self.__class__(
            series.rename(series.name, copy=False),
            implementation=self._implementation,
        )

    @property
    def name(self) -> str:
        return self._name  # type: ignore[no-any-return]

    @property
    def series(self) -> Any:
        return self._series

    def cast(
        self,
        dtype: DType,  # type: ignore[override]
    ) -> Self:
        ser = self.series
        dtype = reverse_translate_dtype(dtype)
        return self._from_series(ser.astype(dtype))

    def filter(self, mask: Self) -> Self:
        ser = self.series
        return self._from_series(ser.loc[validate_column_comparand(mask)])

    def item(self) -> Any:
        return item(self.series)

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Series:
        ser = self.series
        return self._from_series(ser.between(lower_bound, upper_bound, inclusive=closed))

    def is_in(self, other: Any) -> Series:
        ser = self.series
        return self._from_series(ser.isin(other))

    # Binary comparisons

    def __eq__(self, other: object) -> Series:  # type: ignore[override]
        other = validate_column_comparand(other)
        ser = self.series
        return self._from_series((ser == other).rename(ser.name, copy=False))

    def __ne__(self, other: object) -> Series:  # type: ignore[override]
        other = validate_column_comparand(other)
        ser = self.series
        return self._from_series((ser != other).rename(ser.name, copy=False))

    def __ge__(self, other: Any) -> Series:
        other = validate_column_comparand(other)
        ser = self.series
        return self._from_series((ser >= other).rename(ser.name, copy=False))

    def __gt__(self, other: Any) -> Series:
        other = validate_column_comparand(other)
        ser = self.series
        return self._from_series((ser > other).rename(ser.name, copy=False))

    def __le__(self, other: Any) -> Series:
        other = validate_column_comparand(other)
        ser = self.series
        return self._from_series((ser <= other).rename(ser.name, copy=False))

    def __lt__(self, other: Any) -> Series:
        other = validate_column_comparand(other)
        ser = self.series
        return self._from_series((ser < other).rename(ser.name, copy=False))

    def __and__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser & other).rename(ser.name, copy=False))

    def __rand__(self, other: Any) -> Series:
        return self.__and__(other)

    def __or__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser | other).rename(ser.name, copy=False))

    def __ror__(self, other: Any) -> Series:
        return self.__or__(other)

    def __add__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser + other).rename(ser.name, copy=False))

    def __radd__(self, other: Any) -> Series:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser - other).rename(ser.name, copy=False))

    def __rsub__(self, other: Any) -> Series:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser * other).rename(ser.name, copy=False))

    def __rmul__(self, other: Any) -> Series:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser / other).rename(ser.name, copy=False))

    def __rtruediv__(self, other: Any) -> Series:
        raise NotImplementedError

    def __floordiv__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser // other).rename(ser.name, copy=False))

    def __rfloordiv__(self, other: Any) -> Series:
        raise NotImplementedError

    def __pow__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
        return self._from_series((ser**other).rename(ser.name, copy=False))

    def __rpow__(self, other: Any) -> Series:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Any) -> Series:
        ser = self.series
        other = validate_column_comparand(other)
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

    def drop_nulls(self) -> Series:
        ser = self.series
        return self._from_series(ser.dropna())

    def n_unique(self) -> int:
        ser = self.series
        return ser.nunique()  # type: ignore[no-any-return]

    def zip_with(self, mask: SeriesProtocol, other: SeriesProtocol) -> Series:
        ser = self.series
        return self._from_series(ser.where(mask, other))

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> Series:
        ser = self.series
        return self._from_series(
            ser.sample(n=n, frac=fraction, with_replacement=with_replacement)
        )

    def unique(self) -> Series:
        ser = self.series
        return ser.unique()  # type: ignore[no-any-return]

    def is_nan(self) -> Series:
        ser = self.series
        if is_extension_array_dtype(ser.dtype):
            return self._from_series((ser != ser).fillna(False))  # noqa: PLR0124
        return self._from_series(ser.isna())

    def sort(
        self,
        *,
        descending: bool | Sequence[bool] = True,
    ) -> Series:
        ser = self.series
        return self._from_series(
            ser.sort_values(ascending=not descending).rename(self.name)
        )

    def alias(self, name: str) -> Self:
        ser = self.series
        return self._from_series(ser.rename(name, copy=False))

    def to_numpy(self) -> Any:
        return self.series.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation == "pandas":
            return self.series
        elif self._implementation == "cudf":
            return self.series.to_pandas()
        elif self._implementation == "modin":
            return self.series._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"
        raise TypeError(msg)

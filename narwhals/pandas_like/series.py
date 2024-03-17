from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from pandas.api.types import is_extension_array_dtype

from narwhals.pandas_like.utils import item
from narwhals.pandas_like.utils import reset_index
from narwhals.pandas_like.utils import reverse_translate_dtype
from narwhals.pandas_like.utils import translate_dtype
from narwhals.pandas_like.utils import validate_column_comparand

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType


class PandasSeries:
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

        self._name = str(series.name) if series.name is not None else ""
        self._series = reset_index(series)
        self._implementation = implementation

    def _from_series(self, series: Any) -> Self:
        return self.__class__(
            series.rename(series.name, copy=False),
            implementation=self._implementation,
        )

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int]:
        return self._series.shape  # type: ignore[no-any-return]

    def rename(self, name: str) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.rename(name, copy=False))

    @property
    def dtype(self) -> DType:
        return translate_dtype(self._series.dtype)

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        ser = self._series
        dtype = reverse_translate_dtype(dtype)
        return self._from_series(ser.astype(dtype))

    def filter(self, mask: Self) -> Self:
        ser = self._series
        return self._from_series(ser.loc[validate_column_comparand(mask)])

    def item(self) -> Any:
        return item(self._series)

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.between(lower_bound, upper_bound, inclusive=closed))

    def is_in(self, other: Any) -> PandasSeries:
        import pandas as pd

        ser = self._series
        res = ser.isin(other).convert_dtypes()
        res[ser.isna()] = pd.NA
        return self._from_series(res)

    # Binary comparisons

    def __eq__(self, other: object) -> PandasSeries:  # type: ignore[override]
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__eq__(other)).rename(ser.name, copy=False))

    def __ne__(self, other: object) -> PandasSeries:  # type: ignore[override]
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__ne__(other)).rename(ser.name, copy=False))

    def __ge__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__ge__(other)).rename(ser.name, copy=False))

    def __gt__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__gt__(other)).rename(ser.name, copy=False))

    def __le__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__le__(other)).rename(ser.name, copy=False))

    def __lt__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__lt__(other)).rename(ser.name, copy=False))

    def __and__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__and__(other)).rename(ser.name, copy=False))

    def __rand__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rand__(other)).rename(ser.name, copy=False))

    def __or__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__or__(other)).rename(ser.name, copy=False))

    def __ror__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__ror__(other)).rename(ser.name, copy=False))

    def __add__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__add__(other)).rename(ser.name, copy=False))

    def __radd__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__radd__(other)).rename(ser.name, copy=False))

    def __sub__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__sub__(other)).rename(ser.name, copy=False))

    def __rsub__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rsub__(other)).rename(ser.name, copy=False))

    def __mul__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__mul__(other)).rename(ser.name, copy=False))

    def __rmul__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rmul__(other)).rename(ser.name, copy=False))

    def __truediv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__truediv__(other)).rename(ser.name, copy=False))

    def __rtruediv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rtruediv__(other)).rename(ser.name, copy=False))

    def __floordiv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__floordiv__(other)).rename(ser.name, copy=False))

    def __rfloordiv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rfloordiv__(other)).rename(ser.name, copy=False))

    def __pow__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__pow__(other)).rename(ser.name, copy=False))

    def __rpow__(self, other: Any) -> PandasSeries:  # pragma: no cover
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rpow__(other)).rename(ser.name, copy=False))

    def __mod__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__mod__(other)).rename(ser.name, copy=False))

    def __rmod__(self, other: Any) -> PandasSeries:  # pragma: no cover
        ser = self._series
        other = validate_column_comparand(other)
        return self._from_series((ser.__rmod__(other)).rename(ser.name, copy=False))

    # Unary

    def __invert__(self: PandasSeries) -> PandasSeries:
        ser = self._series
        return self._from_series(~ser)

    # Reductions

    def any(self) -> Any:
        ser = self._series
        return ser.any()

    def all(self) -> Any:
        ser = self._series
        return ser.all()

    def min(self) -> Any:
        ser = self._series
        return ser.min()

    def max(self) -> Any:
        ser = self._series
        return ser.max()

    def sum(self) -> Any:
        ser = self._series
        return ser.sum()

    def prod(self) -> Any:
        ser = self._series
        return ser.prod()

    def median(self) -> Any:
        ser = self._series
        return ser.median()

    def mean(self) -> Any:
        ser = self._series
        return ser.mean()

    def std(
        self,
        *,
        correction: float = 1.0,
    ) -> Any:
        ser = self._series
        return ser.std(ddof=correction)

    def var(
        self,
        *,
        correction: float = 1.0,
    ) -> Any:
        ser = self._series
        return ser.var(ddof=correction)

    def len(self) -> Any:
        return len(self._series)

    # Transformations

    def is_null(self) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.isna())

    def drop_nulls(self) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.dropna())

    def n_unique(self) -> int:
        ser = self._series
        return ser.nunique()  # type: ignore[no-any-return]

    def zip_with(self, mask: PandasSeries, other: PandasSeries) -> PandasSeries:
        mask = validate_column_comparand(mask)
        other = validate_column_comparand(other)
        ser = self._series
        return self._from_series(ser.where(mask, other))

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> PandasSeries:
        ser = self._series
        return self._from_series(
            ser.sample(n=n, frac=fraction, with_replacement=with_replacement)
        )

    def unique(self) -> PandasSeries:
        if self._implementation != "pandas":
            raise NotImplementedError
        import pandas as pd

        return self._from_series(
            pd.Series(
                self._series.unique(), dtype=self._series.dtype, name=self._series.name
            )
        )

    def is_nan(self) -> PandasSeries:
        ser = self._series
        if is_extension_array_dtype(ser.dtype):
            return self._from_series((ser != ser).fillna(False))  # noqa: PLR0124
        return self._from_series(ser.isna())

    def sort(
        self,
        *,
        descending: bool | Sequence[bool] = True,
    ) -> PandasSeries:
        ser = self._series
        return self._from_series(
            ser.sort_values(ascending=not descending).rename(self.name)
        )

    def alias(self, name: str) -> Self:
        ser = self._series
        return self._from_series(ser.rename(name, copy=False))

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation == "pandas":
            return self._series
        elif self._implementation == "cudf":
            return self._series.to_pandas()
        elif self._implementation == "modin":
            return self._series._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"
        raise TypeError(msg)

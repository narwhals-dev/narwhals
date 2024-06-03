from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Sequence

from narwhals._pandas_like.utils import item
from narwhals._pandas_like.utils import reverse_translate_dtype
from narwhals._pandas_like.utils import to_datetime
from narwhals._pandas_like.utils import translate_dtype
from narwhals._pandas_like.utils import validate_column_comparand
from narwhals.dependencies import get_pandas
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals.dtypes import DType

PANDAS_TO_NUMPY_DTYPE_NO_MISSING = {
    "Int64": "int64",
    "int64[pyarrow]": "int64",
    "Int32": "int32",
    "int32[pyarrow]": "int32",
    "Int16": "int16",
    "int16[pyarrow]": "int16",
    "Int8": "int8",
    "int8[pyarrow]": "int8",
    "UInt64": "uint64",
    "uint64[pyarrow]": "uint64",
    "UInt32": "uint32",
    "uint32[pyarrow]": "uint32",
    "UInt16": "uint16",
    "uint16[pyarrow]": "uint16",
    "UInt8": "uint8",
    "uint8[pyarrow]": "uint8",
    "Float64": "float64",
    "float64[pyarrow]": "float64",
    "Float32": "float32",
    "float32[pyarrow]": "float32",
}
PANDAS_TO_NUMPY_DTYPE_MISSING = {
    "Int64": "float64",
    "int64[pyarrow]": "float64",
    "Int32": "float64",
    "int32[pyarrow]": "float64",
    "Int16": "float64",
    "int16[pyarrow]": "float64",
    "Int8": "float64",
    "int8[pyarrow]": "float64",
    "UInt64": "float64",
    "uint64[pyarrow]": "float64",
    "UInt32": "float64",
    "uint32[pyarrow]": "float64",
    "UInt16": "float64",
    "uint16[pyarrow]": "float64",
    "UInt8": "float64",
    "uint8[pyarrow]": "float64",
    "Float64": "float64",
    "float64[pyarrow]": "float64",
    "Float32": "float32",
    "float32[pyarrow]": "float32",
}


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

        self._name = series.name
        self._series = series
        self._implementation = implementation
        self._use_copy_false = False
        if self._implementation == "pandas":
            pd = get_pandas()

            if parse_version(pd.__version__) < parse_version("3.0.0"):
                self._use_copy_false = True
            else:  # pragma: no cover
                pass
        else:  # pragma: no cover
            pass

    def __narwhals_namespace__(self) -> PandasNamespace:
        from narwhals._pandas_like.namespace import PandasNamespace

        return PandasNamespace(self._implementation)

    def __narwhals_series__(self) -> Self:
        return self

    def __getitem__(self, idx: int) -> Any:
        return self._series.iloc[idx]

    def _rename(self, series: Any, name: str) -> Any:
        if self._use_copy_false:
            return series.rename(name, copy=False)
        return series.rename(name)  # pragma: no cover

    def _from_series(self, series: Any) -> Self:
        return self.__class__(
            series,
            implementation=self._implementation,
        )

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def name(self) -> str:
        return self._name  # type: ignore[no-any-return]

    @property
    def shape(self) -> tuple[int]:
        return self._series.shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> DType:
        return translate_dtype(self._series.dtype)

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        ser = self._series
        dtype = reverse_translate_dtype(dtype, ser.dtype, self._implementation)
        return self._from_series(ser.astype(dtype))

    def item(self) -> Any:
        return item(self._series)

    def to_frame(self) -> Any:
        from narwhals._pandas_like.dataframe import PandasDataFrame

        return PandasDataFrame(
            self._series.to_frame(), implementation=self._implementation
        )

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> PandasSeries:
        ser = self._series
        if closed == "left":
            res = ser.ge(lower_bound) & ser.lt(upper_bound)
        elif closed == "right":
            res = ser.gt(lower_bound) & ser.le(upper_bound)
        elif closed == "none":
            res = ser.gt(lower_bound) & ser.lt(upper_bound)
        elif closed == "both":
            res = ser.ge(lower_bound) & ser.le(upper_bound)
        else:  # pragma: no cover
            raise AssertionError
        return self._from_series(res)

    def is_in(self, other: Any) -> PandasSeries:
        ser = self._series
        res = ser.isin(other)
        return self._from_series(res)

    # Binary comparisons

    def filter(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.loc[other], ser.name))

    def __eq__(self, other: object) -> PandasSeries:  # type: ignore[override]
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__eq__(other), ser.name))

    def __ne__(self, other: object) -> PandasSeries:  # type: ignore[override]
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__ne__(other), ser.name))

    def __ge__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__ge__(other), ser.name))

    def __gt__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__gt__(other), ser.name))

    def __le__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__le__(other), ser.name))

    def __lt__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__lt__(other), ser.name))

    def __and__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__and__(other), ser.name))

    def __rand__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rand__(other), ser.name))

    def __or__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__or__(other), ser.name))

    def __ror__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__ror__(other), ser.name))

    def __add__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__add__(other), ser.name))

    def __radd__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__radd__(other), ser.name))

    def __sub__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__sub__(other), ser.name))

    def __rsub__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rsub__(other), ser.name))

    def __mul__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__mul__(other), ser.name))

    def __rmul__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rmul__(other), ser.name))

    def __truediv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__truediv__(other), ser.name))

    def __rtruediv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rtruediv__(other), ser.name))

    def __floordiv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__floordiv__(other), ser.name))

    def __rfloordiv__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rfloordiv__(other), ser.name))

    def __pow__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__pow__(other), ser.name))

    def __rpow__(self, other: Any) -> PandasSeries:  # pragma: no cover
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rpow__(other), ser.name))

    def __mod__(self, other: Any) -> PandasSeries:
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__mod__(other), ser.name))

    def __rmod__(self, other: Any) -> PandasSeries:  # pragma: no cover
        ser = self._series
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(self._rename(ser.__rmod__(other), ser.name))

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

    def mean(self) -> Any:
        ser = self._series
        return ser.mean()

    def std(
        self,
        *,
        ddof: int = 1,
    ) -> Any:
        ser = self._series
        return ser.std(ddof=ddof)

    def len(self) -> Any:
        return len(self._series)

    # Transformations

    def is_null(self) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.isna())

    def fill_null(self, value: Any) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.fillna(value))

    def drop_nulls(self) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.dropna())

    def n_unique(self) -> int:
        ser = self._series
        return ser.nunique(dropna=False)  # type: ignore[no-any-return]

    def sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> PandasSeries:
        ser = self._series
        return self._from_series(ser.sample(n=n, frac=fraction, replace=with_replacement))

    def cum_sum(self) -> PandasSeries:
        return self._from_series(
            self._series.__class__(
                self._series.cumsum(), dtype=self._series.dtype, name=self._series.name
            )
        )

    def unique(self) -> PandasSeries:
        return self._from_series(
            self._series.__class__(self._series.unique(), name=self._series.name)
        )

    def diff(self) -> PandasSeries:
        return self._from_series(
            self._series.__class__(self._series.diff(), name=self._series.name)
        )

    def shift(self, n: int) -> PandasSeries:
        return self._from_series(
            self._series.__class__(self._series.shift(n), name=self._series.name)
        )

    def sort(
        self,
        *,
        descending: bool | Sequence[bool] = False,
    ) -> PandasSeries:
        ser = self._series
        return self._from_series(
            ser.sort_values(ascending=not descending, na_position="first").rename(
                self.name
            )
        )

    def alias(self, name: str) -> Self:
        ser = self._series
        return self._from_series(self._rename(ser, name))

    def to_numpy(self) -> Any:
        has_missing = self._series.isna().any()
        if has_missing and str(self._series.dtype) in PANDAS_TO_NUMPY_DTYPE_MISSING:
            return self._series.to_numpy(
                dtype=PANDAS_TO_NUMPY_DTYPE_MISSING[str(self._series.dtype)],
                na_value=float("nan"),
            )
        if (
            not has_missing
            and str(self._series.dtype) in PANDAS_TO_NUMPY_DTYPE_NO_MISSING
        ):
            return self._series.to_numpy(
                dtype=PANDAS_TO_NUMPY_DTYPE_NO_MISSING[str(self._series.dtype)]
            )
        return self._series.to_numpy()

    def to_pandas(self) -> Any:
        if self._implementation == "pandas":
            return self._series
        elif self._implementation == "cudf":  # pragma: no cover
            return self._series.to_pandas()
        elif self._implementation == "modin":  # pragma: no cover
            return self._series._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"  # pragma: no cover
        raise AssertionError(msg)

    # --- descriptive ---
    def is_duplicated(self: Self) -> Self:
        return self._from_series(self._series.duplicated(keep=False))

    def is_empty(self: Self) -> bool:
        return self._series.empty  # type: ignore[no-any-return]

    def is_unique(self: Self) -> Self:
        return self._from_series(~self._series.duplicated(keep=False))

    def null_count(self: Self) -> int:
        return self._series.isnull().sum()  # type: ignore[no-any-return]

    def is_first_distinct(self: Self) -> Self:
        return self._from_series(~self._series.duplicated(keep="first"))

    def is_last_distinct(self: Self) -> Self:
        return self._from_series(~self._series.duplicated(keep="last"))

    def is_sorted(self: Self, *, descending: bool = False) -> bool:
        if not isinstance(descending, bool):
            msg = f"argument 'descending' should be boolean, found {type(descending)}"
            raise TypeError(msg)

        if descending:
            return self._series.is_monotonic_decreasing  # type: ignore[no-any-return]
        else:
            return self._series.is_monotonic_increasing  # type: ignore[no-any-return]

    def value_counts(self: Self, *, sort: bool = False, parallel: bool = False) -> Any:
        """Parallel is unused, exists for compatibility"""
        from narwhals._pandas_like.dataframe import PandasDataFrame

        name_ = "index" if self._series.name is None else self._series.name
        val_count = self._series.value_counts(dropna=False, sort=False).reset_index()
        val_count.columns = [name_, "count"]
        if sort:
            val_count = val_count.sort_values(name_)

        return PandasDataFrame(
            val_count,
            implementation=self._implementation,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Any:
        return self._series.quantile(q=quantile, interpolation=interpolation)

    def zip_with(self: Self, mask: Any, other: Any) -> PandasSeries:
        ser = self._series
        res = ser.where(mask._series, other._series)
        return self._from_series(res)

    @property
    def str(self) -> PandasSeriesStringNamespace:
        return PandasSeriesStringNamespace(self)

    @property
    def dt(self) -> PandasSeriesDateTimeNamespace:
        return PandasSeriesDateTimeNamespace(self)


class PandasSeriesStringNamespace:
    def __init__(self, series: PandasSeries) -> None:
        self._series = series

    def ends_with(self, suffix: str) -> PandasSeries:
        return self._series._from_series(
            self._series._series.str.endswith(suffix),
        )

    def head(self, n: int = 5) -> PandasSeries:
        return self._series._from_series(
            self._series._series.str[:n],
        )

    def to_datetime(self, format: str | None = None) -> PandasSeries:  # noqa: A002
        return self._series._from_series(
            to_datetime(self._series._implementation)(self._series._series, format=format)
        )


class PandasSeriesDateTimeNamespace:
    def __init__(self, series: PandasSeries) -> None:
        self._series = series

    def year(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.year,
        )

    def month(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.month,
        )

    def day(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.day,
        )

    def hour(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.hour,
        )

    def minute(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.minute,
        )

    def second(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.second,
        )

    def millisecond(self) -> PandasSeries:
        return self._series._from_series(
            self._series._series.dt.microsecond // 1000,
        )

    def microsecond(self) -> PandasSeries:
        return self._series._from_series(self._series._series.dt.microsecond)

    def nanosecond(self) -> PandasSeries:
        return self._series._from_series(
            (
                (self._series._series.dt.microsecond * 1_000)
                + self._series._series.dt.nanosecond
            ),
        )

    def ordinal_day(self) -> PandasSeries:
        ser = self._series._series
        year_start = ser.dt.year
        result = (
            ser.to_numpy().astype("datetime64[D]")
            - (year_start.to_numpy() - 1970).astype("datetime64[Y]")
        ).astype("int32") + 1
        dtype = "Int64[pyarrow]" if "pyarrow" in str(ser.dtype) else "int32"
        return self._series._from_series(
            self._series._series.__class__(result, dtype=dtype, name=year_start.name)
        )

    def total_minutes(self) -> PandasSeries:
        s = self._series._series.dt.total_seconds()
        s_sign = (
            2 * (s > 0).astype(int) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 60
        if ~s.isna().any():
            s_abs = s_abs.astype(int)
        return self._series._from_series(s_abs * s_sign)

    def total_seconds(self) -> PandasSeries:
        s = self._series._series.dt.total_seconds()
        s_sign = (
            2 * (s > 0).astype(int) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int)
        return self._series._from_series(s_abs * s_sign)

    def total_milliseconds(self) -> PandasSeries:
        s = self._series._series.dt.total_seconds() * 1e3
        s_sign = (
            2 * (s > 0).astype(int) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int)
        return self._series._from_series(s_abs * s_sign)

    def total_microseconds(self) -> PandasSeries:
        s = self._series._series.dt.total_seconds() * 1e6
        s_sign = (
            2 * (s > 0).astype(int) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int)
        return self._series._from_series(s_abs * s_sign)

    def total_nanoseconds(self) -> PandasSeries:
        s = self._series._series.dt.total_seconds() * 1e9
        s_sign = (
            2 * (s > 0).astype(int) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int)
        return self._series._from_series(s_abs * s_sign)

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._pandas_like.utils import calculate_timestamp_date
from narwhals._pandas_like.utils import calculate_timestamp_datetime
from narwhals._pandas_like.utils import int_dtype_mapper
from narwhals._pandas_like.utils import narwhals_to_native_dtype
from narwhals._pandas_like.utils import native_series_from_iterable
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals._pandas_like.utils import set_axis
from narwhals._pandas_like.utils import to_datetime
from narwhals._pandas_like.utils import validate_column_comparand
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes

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


class PandasLikeSeries:
    def __init__(
        self,
        native_series: Any,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._name = native_series.name
        self._native_series = native_series
        self._implementation = implementation
        self._backend_version = backend_version
        self._dtypes = dtypes

        # In pandas, copy-on-write becomes the default in version 3.
        # So, before that, we need to explicitly avoid unnecessary
        # copies by using `copy=False` sometimes.
        if self._implementation is Implementation.PANDAS and self._backend_version < (
            3,
            0,
            0,
        ):
            self._use_copy_false = True
        else:
            self._use_copy_false = False

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation in {
            Implementation.PANDAS,
            Implementation.MODIN,
            Implementation.CUDF,
        }:
            return self._implementation.to_native_namespace()

        msg = f"Expected pandas/modin/cudf, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __narwhals_series__(self) -> Self:
        return self

    @overload
    def __getitem__(self, idx: int) -> Any: ...

    @overload
    def __getitem__(self, idx: slice | Sequence[int]) -> Self: ...

    def __getitem__(self, idx: int | slice | Sequence[int]) -> Any | Self:
        if isinstance(idx, int):
            return self._native_series.iloc[idx]
        return self._from_native_series(self._native_series.iloc[idx])

    def _from_native_series(self, series: Any) -> Self:
        return self.__class__(
            series,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    @classmethod
    def _from_iterable(
        cls: type[Self],
        data: Iterable[Any],
        name: str,
        index: Any,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> Self:
        return cls(
            native_series_from_iterable(
                data,
                name=name,
                index=index,
                implementation=implementation,
            ),
            implementation=implementation,
            backend_version=backend_version,
            dtypes=dtypes,
        )

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def name(self) -> str:
        return self._name  # type: ignore[no-any-return]

    @property
    def shape(self) -> tuple[int]:
        return self._native_series.shape  # type: ignore[no-any-return]

    @property
    def dtype(self: Self) -> DType:
        return native_to_narwhals_dtype(
            self._native_series, self._dtypes, self._implementation
        )

    def scatter(self, indices: int | Sequence[int], values: Any) -> Self:
        if isinstance(values, self.__class__):
            # .copy() is necessary in some pre-2.2 versions of pandas to avoid
            # `values` also getting modified (!)
            values = validate_column_comparand(self._native_series.index, values).copy()
            values = set_axis(
                values,
                self._native_series.index[indices],
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        s = self._native_series.copy(deep=True)
        s.iloc[indices] = values
        s.name = self.name
        return self._from_native_series(s)

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        ser = self._native_series
        dtype = narwhals_to_native_dtype(
            dtype, ser.dtype, self._implementation, self._backend_version, self._dtypes
        )
        return self._from_native_series(ser.astype(dtype))

    def item(self: Self, index: int | None = None) -> Any:
        # cuDF doesn't have Series.item().
        if index is None:
            if len(self) != 1:
                msg = (
                    "can only call '.item()' if the Series is of length 1,"
                    f" or an explicit index is provided (Series is of length {len(self)})"
                )
                raise ValueError(msg)
            return self._native_series.iloc[0]
        return self._native_series.iloc[index]

    def to_frame(self) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        return PandasLikeDataFrame(
            self._native_series.to_frame(),
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def to_list(self) -> Any:
        return self._native_series.to_list()

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> PandasLikeSeries:
        ser = self._native_series
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
        return self._from_native_series(res)

    def is_in(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        res = ser.isin(other)
        return self._from_native_series(res)

    def arg_true(self) -> PandasLikeSeries:
        ser = self._native_series
        result = ser.__class__(range(len(ser)), name=ser.name, index=ser.index).loc[ser]
        return self._from_native_series(result)

    # Binary comparisons

    def filter(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        if not (isinstance(other, list) and all(isinstance(x, bool) for x in other)):
            other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.loc[other].rename(ser.name, copy=False))

    def __eq__(self, other: object) -> PandasLikeSeries:  # type: ignore[override]
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__eq__(other).rename(ser.name, copy=False))

    def __ne__(self, other: object) -> PandasLikeSeries:  # type: ignore[override]
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__ne__(other).rename(ser.name, copy=False))

    def __ge__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__ge__(other).rename(ser.name, copy=False))

    def __gt__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__gt__(other).rename(ser.name, copy=False))

    def __le__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__le__(other).rename(ser.name, copy=False))

    def __lt__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__lt__(other).rename(ser.name, copy=False))

    def __and__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__and__(other).rename(ser.name, copy=False))

    def __rand__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__rand__(other).rename(ser.name, copy=False))

    def __or__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__or__(other).rename(ser.name, copy=False))

    def __ror__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__ror__(other).rename(ser.name, copy=False))

    def __add__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__add__(other).rename(ser.name, copy=False))

    def __radd__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__radd__(other).rename(ser.name, copy=False))

    def __sub__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__sub__(other).rename(ser.name, copy=False))

    def __rsub__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__rsub__(other).rename(ser.name, copy=False))

    def __mul__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__mul__(other).rename(ser.name, copy=False))

    def __rmul__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__rmul__(other).rename(ser.name, copy=False))

    def __truediv__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(
            ser.__truediv__(other).rename(ser.name, copy=False)
        )

    def __rtruediv__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(
            ser.__rtruediv__(other).rename(ser.name, copy=False)
        )

    def __floordiv__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(
            ser.__floordiv__(other).rename(ser.name, copy=False)
        )

    def __rfloordiv__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(
            ser.__rfloordiv__(other).rename(ser.name, copy=False)
        )

    def __pow__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__pow__(other).rename(ser.name, copy=False))

    def __rpow__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__rpow__(other).rename(ser.name, copy=False))

    def __mod__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__mod__(other).rename(ser.name, copy=False))

    def __rmod__(self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        other = validate_column_comparand(self._native_series.index, other)
        return self._from_native_series(ser.__rmod__(other).rename(ser.name, copy=False))

    # Unary

    def __invert__(self: PandasLikeSeries) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(~ser)

    # Reductions

    def any(self) -> Any:
        ser = self._native_series
        return ser.any()

    def all(self) -> Any:
        ser = self._native_series
        return ser.all()

    def min(self) -> Any:
        ser = self._native_series
        return ser.min()

    def max(self) -> Any:
        ser = self._native_series
        return ser.max()

    def sum(self) -> Any:
        ser = self._native_series
        return ser.sum()

    def count(self) -> Any:
        ser = self._native_series
        return ser.count()

    def mean(self) -> Any:
        ser = self._native_series
        return ser.mean()

    def std(
        self,
        *,
        ddof: int = 1,
    ) -> Any:
        ser = self._native_series
        return ser.std(ddof=ddof)

    def len(self) -> Any:
        return len(self._native_series)

    # Transformations

    def is_null(self) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(ser.isna())

    def fill_null(self, value: Any) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(ser.fillna(value))

    def drop_nulls(self) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(ser.dropna())

    def n_unique(self) -> int:
        ser = self._native_series
        return ser.nunique(dropna=False)  # type: ignore[no-any-return]

    def sample(
        self: Self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        ser = self._native_series
        return self._from_native_series(
            ser.sample(n=n, frac=fraction, replace=with_replacement, random_state=seed)
        )

    def abs(self) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.abs())

    def cum_sum(self) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.cumsum())

    def unique(self) -> PandasLikeSeries:
        return self._from_native_series(
            self._native_series.__class__(
                self._native_series.unique(), name=self._native_series.name
            )
        )

    def diff(self) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.diff())

    def shift(self, n: int) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.shift(n))

    def sort(
        self, *, descending: bool = False, nulls_last: bool = False
    ) -> PandasLikeSeries:
        ser = self._native_series
        na_position = "last" if nulls_last else "first"
        return self._from_native_series(
            ser.sort_values(ascending=not descending, na_position=na_position).rename(
                self.name, copy=False
            )
        )

    def alias(self, name: str) -> Self:
        if name != self.name:
            ser = self._native_series
            return self._from_native_series(ser.rename(name, copy=False))
        return self

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        # pandas used to always return object dtype for nullable dtypes.
        # So, we intercept __array__ and pass to `to_numpy` ourselves to make
        # sure an appropriate numpy dtype is returned.
        return self.to_numpy(dtype=dtype, copy=copy)

    def to_numpy(self, dtype: Any = None, copy: bool | None = None) -> Any:
        # the default is meant to be None, but pandas doesn't allow it?
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__array__.html
        copy = copy or self._implementation is Implementation.CUDF
        if self.dtype == self._dtypes.Datetime and self.dtype.time_zone is not None:  # type: ignore[attr-defined]
            s = self.dt.convert_time_zone("UTC").dt.replace_time_zone(None)._native_series
        else:
            s = self._native_series

        has_missing = s.isna().any()
        if has_missing and str(s.dtype) in PANDAS_TO_NUMPY_DTYPE_MISSING:
            if self._implementation is Implementation.PANDAS and self._backend_version < (
                1,
            ):  # pragma: no cover
                kwargs = {}
            else:
                kwargs = {"na_value": float("nan")}
            return s.to_numpy(
                dtype=dtype or PANDAS_TO_NUMPY_DTYPE_MISSING[str(s.dtype)],
                copy=copy,
                **kwargs,
            )
        if not has_missing and str(s.dtype) in PANDAS_TO_NUMPY_DTYPE_NO_MISSING:
            return s.to_numpy(
                dtype=dtype or PANDAS_TO_NUMPY_DTYPE_NO_MISSING[str(s.dtype)],
                copy=copy,
            )
        return s.to_numpy(dtype=dtype, copy=copy)

    def to_pandas(self) -> Any:
        if self._implementation is Implementation.PANDAS:
            return self._native_series
        elif self._implementation is Implementation.CUDF:  # pragma: no cover
            return self._native_series.to_pandas()
        elif self._implementation is Implementation.MODIN:  # pragma: no cover
            return self._native_series._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"  # pragma: no cover
        raise AssertionError(msg)

    # --- descriptive ---
    def is_duplicated(self: Self) -> Self:
        res = self._native_series.duplicated(keep=False)
        res = res.rename(self.name, copy=False)
        return self._from_native_series(res)

    def is_empty(self: Self) -> bool:
        return self._native_series.empty  # type: ignore[no-any-return]

    def is_unique(self: Self) -> Self:
        res = ~self._native_series.duplicated(keep=False).rename(self.name, copy=False)
        return self._from_native_series(res)

    def null_count(self: Self) -> int:
        return self._native_series.isna().sum()  # type: ignore[no-any-return]

    def is_first_distinct(self: Self) -> Self:
        res = ~self._native_series.duplicated(keep="first").rename(self.name, copy=False)
        return self._from_native_series(res)

    def is_last_distinct(self: Self) -> Self:
        res = ~self._native_series.duplicated(keep="last").rename(self.name, copy=False)
        return self._from_native_series(res)

    def is_sorted(self: Self, *, descending: bool = False) -> bool:
        if not isinstance(descending, bool):
            msg = f"argument 'descending' should be boolean, found {type(descending)}"
            raise TypeError(msg)

        if descending:
            return self._native_series.is_monotonic_decreasing  # type: ignore[no-any-return]
        else:
            return self._native_series.is_monotonic_increasing  # type: ignore[no-any-return]

    def value_counts(
        self: Self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> PandasLikeDataFrame:
        """Parallel is unused, exists for compatibility"""
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        index_name_ = "index" if self._name is None else self._name
        value_name_ = name or ("proportion" if normalize else "count")

        val_count = self._native_series.value_counts(
            dropna=False,
            sort=False,
            normalize=normalize,
        ).reset_index()

        val_count.columns = [index_name_, value_name_]

        if sort:
            val_count = val_count.sort_values(value_name_, ascending=False)

        return PandasLikeDataFrame(
            val_count,
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Any:
        return self._native_series.quantile(q=quantile, interpolation=interpolation)

    def zip_with(self: Self, mask: Any, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        mask = validate_column_comparand(ser.index, mask)
        other = validate_column_comparand(ser.index, other)
        res = ser.where(mask, other)
        return self._from_native_series(res)

    def head(self: Self, n: int) -> Self:
        return self._from_native_series(self._native_series.head(n))

    def tail(self: Self, n: int) -> Self:
        return self._from_native_series(self._native_series.tail(n))

    def round(self: Self, decimals: int) -> Self:
        return self._from_native_series(self._native_series.round(decimals=decimals))

    def to_dummies(
        self: Self, *, separator: str = "_", drop_first: bool = False
    ) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        plx = self.__native_namespace__()
        series = self._native_series
        name = str(self._name) if self._name else ""
        return PandasLikeDataFrame(
            plx.get_dummies(
                series,
                prefix=name,
                prefix_sep=separator,
                drop_first=drop_first,
            ).astype(int),
            implementation=self._implementation,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_native_series(self._native_series.iloc[offset::n])

    def clip(
        self: Self, lower_bound: Any | None = None, upper_bound: Any | None = None
    ) -> Self:
        return self._from_native_series(
            self._native_series.clip(lower_bound, upper_bound)
        )

    def to_arrow(self: Self) -> Any:
        if self._implementation is Implementation.CUDF:  # pragma: no cover
            return self._native_series.to_arrow()

        import pyarrow as pa  # ignore-banned-import()

        return pa.Array.from_pandas(self._native_series)

    def mode(self: Self) -> Self:
        native_series = self._native_series
        result = native_series.mode()
        result.name = native_series.name
        return self._from_native_series(result)

    def __iter__(self: Self) -> Iterator[Any]:
        yield from self._native_series.__iter__()

    @property
    def str(self) -> PandasLikeSeriesStringNamespace:
        return PandasLikeSeriesStringNamespace(self)

    @property
    def dt(self) -> PandasLikeSeriesDateTimeNamespace:
        return PandasLikeSeriesDateTimeNamespace(self)

    @property
    def cat(self) -> PandasLikeSeriesCatNamespace:
        return PandasLikeSeriesCatNamespace(self)


class PandasLikeSeriesCatNamespace:
    def __init__(self, series: PandasLikeSeries) -> None:
        self._pandas_series = series

    def get_categories(self) -> PandasLikeSeries:
        s = self._pandas_series._native_series
        return self._pandas_series._from_native_series(
            s.__class__(s.cat.categories, name=s.name)
        )


class PandasLikeSeriesStringNamespace:
    def __init__(self, series: PandasLikeSeries) -> None:
        self._pandas_series = series

    def len_chars(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.len()
        )

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.replace(
                pat=pattern, repl=value, n=n, regex=not literal
            ),
        )

    def replace_all(
        self, pattern: str, value: str, *, literal: bool = False
    ) -> PandasLikeSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self, characters: str | None) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.strip(characters),
        )

    def starts_with(self, prefix: str) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.startswith(prefix),
        )

    def ends_with(self, suffix: str) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.endswith(suffix),
        )

    def contains(self, pattern: str, *, literal: bool = False) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.contains(
                pat=pattern, regex=not literal
            )
        )

    def slice(self, offset: int, length: int | None = None) -> PandasLikeSeries:
        stop = offset + length if length else None
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.slice(start=offset, stop=stop),
        )

    def to_datetime(self: Self, format: str | None) -> PandasLikeSeries:  # noqa: A002
        return self._pandas_series._from_native_series(
            to_datetime(self._pandas_series._implementation)(
                self._pandas_series._native_series, format=format
            )
        )

    def to_uppercase(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.upper(),
        )

    def to_lowercase(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.str.lower(),
        )


class PandasLikeSeriesDateTimeNamespace:
    def __init__(self, series: PandasLikeSeries) -> None:
        self._pandas_series = series

    def date(self) -> PandasLikeSeries:
        result = self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.date,
        )
        if str(result.dtype).lower() == "object":
            msg = (
                "Accessing `date` on the default pandas backend "
                "will return a Series of type `object`."
                "\nThis differs from polars API and will prevent `.dt` chaining. "
                "Please switch to the `pyarrow` backend:"
                '\ndf.convert_dtypes(dtype_backend="pyarrow")'
            )
            raise NotImplementedError(msg)
        return result

    def year(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.year,
        )

    def month(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.month,
        )

    def day(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.day,
        )

    def hour(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.hour,
        )

    def minute(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.minute,
        )

    def second(self) -> PandasLikeSeries:
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.second,
        )

    def millisecond(self) -> PandasLikeSeries:
        return self.microsecond() // 1000

    def microsecond(self) -> PandasLikeSeries:
        if self._pandas_series._backend_version < (3, 0, 0) and "pyarrow" in str(
            self._pandas_series._native_series.dtype
        ):
            # crazy workaround for https://github.com/pandas-dev/pandas/issues/59154
            import pyarrow.compute as pc  # ignore-banned-import()

            native_series = self._pandas_series._native_series
            arr = native_series.array.__arrow_array__()
            result_arr = pc.add(
                pc.multiply(pc.millisecond(arr), 1000), pc.microsecond(arr)
            )
            result = native_series.__class__(
                native_series.array.__class__(result_arr), name=native_series.name
            )
            return self._pandas_series._from_native_series(result)

        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.microsecond
        )

    def nanosecond(self) -> PandasLikeSeries:
        return (  # type: ignore[no-any-return]
            self.microsecond() * 1_000 + self._pandas_series._native_series.dt.nanosecond
        )

    def ordinal_day(self) -> PandasLikeSeries:
        ser = self._pandas_series._native_series
        year_start = ser.dt.year
        result = (
            ser.to_numpy().astype("datetime64[D]")
            - (year_start.to_numpy() - 1970).astype("datetime64[Y]")
        ).astype("int32") + 1
        dtype = "Int64[pyarrow]" if "pyarrow" in str(ser.dtype) else "int32"
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.__class__(
                result, dtype=dtype, name=year_start.name
            )
        )

    def _get_total_seconds(self) -> Any:
        if hasattr(self._pandas_series._native_series.dt, "total_seconds"):
            return self._pandas_series._native_series.dt.total_seconds()
        else:  # pragma: no cover
            return (
                self._pandas_series._native_series.dt.days * 86400
                + self._pandas_series._native_series.dt.seconds
                + (self._pandas_series._native_series.dt.microseconds / 1e6)
                + (self._pandas_series._native_series.dt.nanoseconds / 1e9)
            )

    def total_minutes(self) -> PandasLikeSeries:
        s = self._get_total_seconds()
        s_sign = (
            2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 60
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self._pandas_series._from_native_series(s_abs * s_sign)

    def total_seconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds()
        s_sign = (
            2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self._pandas_series._from_native_series(s_abs * s_sign)

    def total_milliseconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds() * 1e3
        s_sign = (
            2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self._pandas_series._from_native_series(s_abs * s_sign)

    def total_microseconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds() * 1e6
        s_sign = (
            2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self._pandas_series._from_native_series(s_abs * s_sign)

    def total_nanoseconds(self) -> PandasLikeSeries:
        s = self._get_total_seconds() * 1e9
        s_sign = (
            2 * (s > 0).astype(int_dtype_mapper(s.dtype)) - 1
        )  # this calculates the sign of each series element
        s_abs = s.abs() // 1
        if ~s.isna().any():
            s_abs = s_abs.astype(int_dtype_mapper(s.dtype))
        return self._pandas_series._from_native_series(s_abs * s_sign)

    def to_string(self, format: str) -> PandasLikeSeries:  # noqa: A002
        # Polars' parser treats `'%.f'` as pandas does `'.%f'`
        # PyArrow interprets `'%S'` as "seconds, plus fractional seconds"
        # and doesn't support `%f`
        if "pyarrow" not in str(self._pandas_series._native_series.dtype):
            format = format.replace("%S%.f", "%S.%f")
        else:
            format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self._pandas_series._from_native_series(
            self._pandas_series._native_series.dt.strftime(format)
        )

    def replace_time_zone(self, time_zone: str | None) -> PandasLikeSeries:
        if time_zone is not None:
            result = self._pandas_series._native_series.dt.tz_localize(
                None
            ).dt.tz_localize(time_zone)
        else:
            result = self._pandas_series._native_series.dt.tz_localize(None)
        return self._pandas_series._from_native_series(result)

    def convert_time_zone(self, time_zone: str) -> PandasLikeSeries:
        if self._pandas_series.dtype.time_zone is None:  # type: ignore[attr-defined]
            result = self._pandas_series._native_series.dt.tz_localize(
                "UTC"
            ).dt.tz_convert(time_zone)
        else:
            result = self._pandas_series._native_series.dt.tz_convert(time_zone)
        return self._pandas_series._from_native_series(result)

    def timestamp(self, time_unit: Literal["ns", "us", "ms"] = "us") -> PandasLikeSeries:
        s = self._pandas_series._native_series
        dtype = self._pandas_series.dtype
        is_pyarrow_dtype = "pyarrow" in str(self._pandas_series._native_series.dtype)
        mask_na = s.isna()
        if dtype == self._pandas_series._dtypes.Date:
            # Date is only supported in pandas dtypes if pyarrow-backed
            s_cast = s.astype("Int32[pyarrow]")
            result = calculate_timestamp_date(s_cast, time_unit)
        elif dtype == self._pandas_series._dtypes.Datetime:
            original_time_unit = dtype.time_unit  # type: ignore[attr-defined]
            if (
                self._pandas_series._implementation is Implementation.PANDAS
                and self._pandas_series._backend_version < (2,)
            ):  # pragma: no cover
                s_cast = s.view("Int64[pyarrow]") if is_pyarrow_dtype else s.view("int64")
            else:
                s_cast = (
                    s.astype("Int64[pyarrow]") if is_pyarrow_dtype else s.astype("int64")
                )
            result = calculate_timestamp_datetime(s_cast, original_time_unit, time_unit)
        else:
            msg = "Input should be either of Date or Datetime type"
            raise TypeError(msg)
        result[mask_na] = None
        return self._pandas_series._from_native_series(result)

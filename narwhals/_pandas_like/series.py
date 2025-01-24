from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._pandas_like.series_cat import PandasLikeSeriesCatNamespace
from narwhals._pandas_like.series_dt import PandasLikeSeriesDateTimeNamespace
from narwhals._pandas_like.series_list import PandasLikeSeriesListNamespace
from narwhals._pandas_like.series_str import PandasLikeSeriesStringNamespace
from narwhals._pandas_like.utils import broadcast_align_and_extract_native
from narwhals._pandas_like.utils import get_dtype_backend
from narwhals._pandas_like.utils import narwhals_to_native_dtype
from narwhals._pandas_like.utils import native_series_from_iterable
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals._pandas_like.utils import rename
from narwhals._pandas_like.utils import select_columns_by_name
from narwhals._pandas_like.utils import set_index
from narwhals.dependencies import is_numpy_scalar
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantSeries
from narwhals.utils import Implementation
from narwhals.utils import import_dtypes_module
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals.dtypes import DType
    from narwhals.utils import Version

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


class PandasLikeSeries(CompliantSeries):
    def __init__(
        self: Self,
        native_series: Any,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._name = native_series.name
        self._native_series = native_series
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation in {
            Implementation.PANDAS,
            Implementation.MODIN,
            Implementation.CUDF,
        }:
            return self._implementation.to_native_namespace()

        msg = f"Expected pandas/modin/cudf, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __narwhals_series__(self: Self) -> Self:
        return self

    @overload
    def __getitem__(self: Self, idx: int) -> Any: ...

    @overload
    def __getitem__(self: Self, idx: slice | Sequence[int]) -> Self: ...

    def __getitem__(self: Self, idx: int | slice | Sequence[int]) -> Any | Self:
        if isinstance(idx, int) or is_numpy_scalar(idx):
            return self._native_series.iloc[idx]
        return self._from_native_series(self._native_series.iloc[idx])

    def _change_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self._native_series,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=version,
        )

    def _from_native_series(self: Self, series: Any) -> Self:
        return self.__class__(
            series,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
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
        version: Version,
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
            version=version,
        )

    def __len__(self: Self) -> int:
        return self.shape[0]

    @property
    def name(self: Self) -> str:
        return self._name  # type: ignore[no-any-return]

    @property
    def shape(self: Self) -> tuple[int]:
        return self._native_series.shape  # type: ignore[no-any-return]

    @property
    def dtype(self: Self) -> DType:
        return native_to_narwhals_dtype(
            self._native_series, self._version, self._implementation
        )

    def ewm_mean(
        self: Self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_periods: int,
        ignore_nulls: bool,
    ) -> PandasLikeSeries:
        ser = self._native_series
        mask_na = ser.isna()
        if self._implementation is Implementation.CUDF:
            if (min_periods == 0 and not ignore_nulls) or (not mask_na.any()):
                result = ser.ewm(
                    com=com, span=span, halflife=half_life, alpha=alpha, adjust=adjust
                ).mean()
            else:
                msg = (
                    "cuDF only supports `ewm_mean` when there are no missing values "
                    "or when both `min_period=0` and `ignore_nulls=False`"
                )
                raise NotImplementedError(msg)
        else:
            result = ser.ewm(
                com, span, half_life, alpha, min_periods, adjust, ignore_na=ignore_nulls
            ).mean()
        result[mask_na] = None
        return self._from_native_series(result)

    def scatter(self: Self, indices: int | Sequence[int], values: Any) -> Self:
        if isinstance(values, self.__class__):
            # .copy() is necessary in some pre-2.2 versions of pandas to avoid
            # `values` also getting modified (!)
            _, values = broadcast_align_and_extract_native(self, values)
            values = set_index(
                values.copy(),
                self._native_series.index[indices],
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        s = self._native_series.copy(deep=True)
        s.iloc[indices] = values
        s.name = self.name
        return self._from_native_series(s)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        ser = self._native_series
        dtype_backend = get_dtype_backend(
            dtype=ser.dtype, implementation=self._implementation
        )
        dtype = narwhals_to_native_dtype(
            dtype,
            dtype_backend=dtype_backend,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )
        return self._from_native_series(ser.astype(dtype))

    def item(self: Self, index: int | None) -> Any:
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

    def to_frame(self: Self) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        return PandasLikeDataFrame(
            self._native_series.to_frame(),
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    def to_list(self: Self) -> list[Any]:
        if self._implementation is Implementation.CUDF:
            return self._native_series.to_arrow().to_pylist()  # type: ignore[no-any-return]
        return self._native_series.to_list()  # type: ignore[no-any-return]

    def is_between(
        self: Self,
        lower_bound: Any,
        upper_bound: Any,
        closed: Literal["left", "right", "none", "both"],
    ) -> PandasLikeSeries:
        ser = self._native_series
        _, lower_bound = broadcast_align_and_extract_native(self, lower_bound)
        _, upper_bound = broadcast_align_and_extract_native(self, upper_bound)
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
        return self._from_native_series(res).alias(ser.name)

    def is_in(self: Self, other: Any) -> PandasLikeSeries:
        ser = self._native_series
        if isinstance(other, self.__class__):
            # We can't use `broadcast_and_align` because we don't want to align here.
            # `other` is just a sequence that all rows from `self` are checked against.
            other = other._native_series
        res = ser.isin(other)
        return self._from_native_series(res)

    def arg_true(self: Self) -> PandasLikeSeries:
        ser = self._native_series
        result = ser.__class__(range(len(ser)), name=ser.name, index=ser.index).loc[ser]
        return self._from_native_series(result)

    def arg_min(self: Self) -> int:
        ser = self._native_series
        if self._implementation is Implementation.PANDAS and self._backend_version < (1,):
            return ser.to_numpy().argmin()  # type: ignore[no-any-return]
        return ser.argmin()  # type: ignore[no-any-return]

    def arg_max(self: Self) -> int:
        ser = self._native_series
        if self._implementation is Implementation.PANDAS and self._backend_version < (1,):
            return ser.to_numpy().argmax()  # type: ignore[no-any-return]
        return ser.argmax()  # type: ignore[no-any-return]

    # Binary comparisons

    def filter(self: Self, other: Any) -> PandasLikeSeries:
        if not (isinstance(other, list) and all(isinstance(x, bool) for x in other)):
            ser, other = broadcast_align_and_extract_native(self, other)
        else:
            ser = self._native_series
        return self._from_native_series(ser.loc[other]).alias(ser.name)

    def __eq__(self: Self, other: object) -> PandasLikeSeries:  # type: ignore[override]
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__eq__(other),
        ).alias(ser.name)

    def __ne__(self: Self, other: object) -> PandasLikeSeries:  # type: ignore[override]
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__ne__(other),
        ).alias(ser.name)

    def __ge__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__ge__(other),
        ).alias(ser.name)

    def __gt__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__gt__(other),
        ).alias(ser.name)

    def __le__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__le__(other),
        ).alias(ser.name)

    def __lt__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__lt__(other),
        ).alias(ser.name)

    def __and__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__and__(other),
        ).alias(ser.name)

    def __rand__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__and__(other),
        ).alias(ser.name)

    def __or__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__or__(other),
        ).alias(ser.name)

    def __ror__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__or__(other),
        ).alias(ser.name)

    def __add__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__add__(other),
        ).alias(ser.name)

    def __radd__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__radd__(other),
        ).alias(ser.name)

    def __sub__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__sub__(other),
        ).alias(ser.name)

    def __rsub__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__rsub__(other),
        ).alias(ser.name)

    def __mul__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__mul__(other),
        ).alias(ser.name)

    def __rmul__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__rmul__(other),
        ).alias(ser.name)

    def __truediv__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__truediv__(other),
        ).alias(ser.name)

    def __rtruediv__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__rtruediv__(other),
        ).alias(ser.name)

    def __floordiv__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__floordiv__(other),
        ).alias(ser.name)

    def __rfloordiv__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__rfloordiv__(other),
        ).alias(ser.name)

    def __pow__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__pow__(other),
        ).alias(ser.name)

    def __rpow__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__rpow__(other),
        ).alias(ser.name)

    def __mod__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__mod__(other),
        ).alias(ser.name)

    def __rmod__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = broadcast_align_and_extract_native(self, other)
        return self._from_native_series(
            ser.__rmod__(other),
        ).alias(ser.name)

    # Unary

    def __invert__(self: PandasLikeSeries) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(~ser)

    # Reductions

    def any(self: Self) -> bool:
        ser = self._native_series
        return ser.any()  # type: ignore[no-any-return]

    def all(self: Self) -> bool:
        ser = self._native_series
        return ser.all()  # type: ignore[no-any-return]

    def min(self: Self) -> Any:
        ser = self._native_series
        return ser.min()

    def max(self: Self) -> Any:
        ser = self._native_series
        return ser.max()

    def sum(self: Self) -> float:
        ser = self._native_series
        return ser.sum()  # type: ignore[no-any-return]

    def count(self: Self) -> int:
        ser = self._native_series
        return ser.count()  # type: ignore[no-any-return]

    def mean(self: Self) -> float:
        ser = self._native_series
        return ser.mean()  # type: ignore[no-any-return]

    def median(self: Self) -> float:
        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)
        ser = self._native_series
        return ser.median()  # type: ignore[no-any-return]

    def std(self: Self, *, ddof: int) -> float:
        ser = self._native_series
        return ser.std(ddof=ddof)  # type: ignore[no-any-return]

    def var(self: Self, *, ddof: int) -> float:
        ser = self._native_series
        return ser.var(ddof=ddof)  # type: ignore[no-any-return]

    def skew(self: Self) -> float | None:
        ser = self._native_series
        ser_not_null = ser.dropna()
        if len(ser_not_null) == 0:
            return None
        elif len(ser_not_null) == 1:
            return float("nan")
        elif len(ser_not_null) == 2:
            return 0.0
        else:
            m = ser_not_null - ser_not_null.mean()
            m2 = (m**2).mean()
            m3 = (m**3).mean()
            return m3 / (m2**1.5) if m2 != 0 else float("nan")

    def len(self: Self) -> int:
        return len(self._native_series)

    # Transformations

    def is_null(self: Self) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(ser.isna())

    def is_nan(self: Self) -> PandasLikeSeries:
        ser = self._native_series
        if self.dtype.is_numeric():
            return self._from_native_series(ser != ser)  # noqa: PLR0124
        msg = f"`.is_nan` only supported for numeric dtype and not {self.dtype}, did you mean `.is_null`?"
        raise InvalidOperationError(msg)

    def fill_null(
        self: Self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self:
        ser = self._native_series
        if value is not None:
            res_ser = self._from_native_series(ser.fillna(value=value))
        else:
            res_ser = self._from_native_series(
                ser.ffill(limit=limit)
                if strategy == "forward"
                else ser.bfill(limit=limit)
            )

        return res_ser

    def drop_nulls(self: Self) -> PandasLikeSeries:
        ser = self._native_series
        return self._from_native_series(ser.dropna())

    def n_unique(self: Self) -> int:
        ser = self._native_series
        return ser.nunique(dropna=False)  # type: ignore[no-any-return]

    def sample(
        self: Self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self:
        ser = self._native_series
        return self._from_native_series(
            ser.sample(n=n, frac=fraction, replace=with_replacement, random_state=seed)
        )

    def abs(self: Self) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.abs())

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        native_series = self._native_series
        result = (
            native_series.cumsum(skipna=True)
            if not reverse
            else native_series[::-1].cumsum(skipna=True)[::-1]
        )
        return self._from_native_series(result)

    def unique(self: Self, *, maintain_order: bool) -> PandasLikeSeries:
        # pandas always maintains order, as per its docstring:
        # "Uniques are returned in order of appearance"  # noqa: ERA001
        return self._from_native_series(
            self._native_series.__class__(
                self._native_series.unique(), name=self._native_series.name
            )
        )

    def diff(self: Self) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.diff())

    def shift(self: Self, n: int) -> PandasLikeSeries:
        return self._from_native_series(self._native_series.shift(n))

    def replace_strict(
        self: Self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> PandasLikeSeries:
        tmp_name = f"{self.name}_tmp"
        dtype = (
            narwhals_to_native_dtype(
                return_dtype,
                self._native_series.dtype,
                self._implementation,
                self._backend_version,
                self._version,
            )
            if return_dtype
            else None
        )
        other = self.__native_namespace__().DataFrame(
            {
                self.name: old,
                tmp_name: self.__native_namespace__().Series(new, dtype=dtype),
            }
        )
        result = self._from_native_series(
            self._native_series.to_frame().merge(other, on=self.name, how="left")[
                tmp_name
            ]
        ).alias(self.name)
        if result.is_null().sum() != self.is_null().sum():
            msg = (
                "replace_strict did not replace all non-null values.\n\n"
                f"The following did not get replaced: {self.filter(~self.is_null() & result.is_null()).unique(maintain_order=False).to_list()}"
            )
            raise ValueError(msg)
        return result

    def sort(self: Self, *, descending: bool, nulls_last: bool) -> PandasLikeSeries:
        ser = self._native_series
        na_position = "last" if nulls_last else "first"
        return self._from_native_series(
            ser.sort_values(ascending=not descending, na_position=na_position)
        ).alias(self.name)

    def alias(self: Self, name: str) -> Self:
        if name != self.name:
            ser = self._native_series
            return self._from_native_series(
                rename(
                    ser,
                    name,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                )
            )
        return self

    def __array__(self: Self, dtype: Any, copy: bool | None) -> np.ndarray:
        # pandas used to always return object dtype for nullable dtypes.
        # So, we intercept __array__ and pass to `to_numpy` ourselves to make
        # sure an appropriate numpy dtype is returned.
        return self.to_numpy(dtype=dtype, copy=copy)

    def to_numpy(self: Self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        # the default is meant to be None, but pandas doesn't allow it?
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__array__.html
        copy = copy or self._implementation is Implementation.CUDF
        dtypes = import_dtypes_module(self._version)
        if self.dtype == dtypes.Datetime and self.dtype.time_zone is not None:  # type: ignore[attr-defined]
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

    def to_pandas(self: Self) -> pd.Series:
        if self._implementation is Implementation.PANDAS:
            return self._native_series
        elif self._implementation is Implementation.CUDF:  # pragma: no cover
            return self._native_series.to_pandas()
        elif self._implementation is Implementation.MODIN:
            return self._native_series._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"  # pragma: no cover
        raise AssertionError(msg)

    def to_polars(self: Self) -> pl.DataFrame:
        import polars as pl  # ignore-banned-import

        if self._implementation is Implementation.PANDAS:
            return pl.from_pandas(self._native_series)
        elif self._implementation is Implementation.CUDF:  # pragma: no cover
            return pl.from_pandas(self._native_series.to_pandas())
        elif self._implementation is Implementation.MODIN:
            return pl.from_pandas(self._native_series._to_pandas())
        msg = f"Unknown implementation: {self._implementation}"  # pragma: no cover
        raise AssertionError(msg)

    # --- descriptive ---
    def is_duplicated(self: Self) -> Self:
        res = self._native_series.duplicated(keep=False)
        return self._from_native_series(res).alias(self.name)

    def is_empty(self: Self) -> bool:
        return self._native_series.empty  # type: ignore[no-any-return]

    def is_unique(self: Self) -> Self:
        res = ~self._native_series.duplicated(keep=False)
        return self._from_native_series(res).alias(self.name)

    def null_count(self: Self) -> int:
        return self._native_series.isna().sum()  # type: ignore[no-any-return]

    def is_first_distinct(self: Self) -> Self:
        res = ~self._native_series.duplicated(keep="first")
        return self._from_native_series(res).alias(self.name)

    def is_last_distinct(self: Self) -> Self:
        res = ~self._native_series.duplicated(keep="last")
        return self._from_native_series(res).alias(self.name)

    def is_sorted(self: Self, *, descending: bool) -> bool:
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
        sort: bool,
        parallel: bool,
        name: str | None,
        normalize: bool,
    ) -> PandasLikeDataFrame:
        """Parallel is unused, exists for compatibility."""
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
            version=self._version,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> float:
        return self._native_series.quantile(q=quantile, interpolation=interpolation)  # type: ignore[no-any-return]

    def zip_with(self: Self, mask: Any, other: Any) -> PandasLikeSeries:
        ser, mask = broadcast_align_and_extract_native(self, mask)
        _, other = broadcast_align_and_extract_native(self, other)
        res = ser.where(mask, other)
        return self._from_native_series(res)

    def head(self: Self, n: int) -> Self:
        return self._from_native_series(self._native_series.head(n))

    def tail(self: Self, n: int) -> Self:
        return self._from_native_series(self._native_series.tail(n))

    def round(self: Self, decimals: int) -> Self:
        return self._from_native_series(self._native_series.round(decimals=decimals))

    def to_dummies(
        self: Self, *, separator: str, drop_first: bool
    ) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        plx = self.__native_namespace__()
        series = self._native_series
        name = str(self._name) if self._name else ""

        null_col_pl = f"{name}{separator}null"

        has_nulls = series.isna().any()
        result = plx.get_dummies(
            series,
            prefix=name,
            prefix_sep=separator,
            drop_first=drop_first,
            # Adds a null column at the end, depending on whether or not there are any.
            dummy_na=has_nulls,
            dtype="int8",
        )
        if has_nulls:
            *cols, null_col_pd = list(result.columns)
            output_order = [null_col_pd, *cols]
            result = rename(
                select_columns_by_name(
                    result, output_order, self._backend_version, self._implementation
                ),
                columns={null_col_pd: null_col_pl},
                implementation=self._implementation,
                backend_version=self._backend_version,
            )

        return PandasLikeDataFrame(
            result,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    def gather_every(self: Self, n: int, offset: int) -> Self:
        return self._from_native_series(self._native_series.iloc[offset::n])

    def clip(
        self: Self, lower_bound: Self | Any | None, upper_bound: Self | Any | None
    ) -> Self:
        _, lower_bound = broadcast_align_and_extract_native(self, lower_bound)
        _, upper_bound = broadcast_align_and_extract_native(self, upper_bound)
        kwargs = {"axis": 0} if self._implementation is Implementation.MODIN else {}
        return self._from_native_series(
            self._native_series.clip(lower_bound, upper_bound, **kwargs)
        )

    def to_arrow(self: Self) -> pa.Array:
        if self._implementation is Implementation.CUDF:
            return self._native_series.to_arrow()

        import pyarrow as pa  # ignore-banned-import()

        return pa.Array.from_pandas(self._native_series)

    def mode(self: Self) -> Self:
        native_series = self._native_series
        result = native_series.mode()
        result.name = native_series.name
        return self._from_native_series(result)

    def cum_count(self: Self, *, reverse: bool) -> Self:
        not_na_series = ~self._native_series.isna()
        result = (
            not_na_series.cumsum()
            if not reverse
            else len(self) - not_na_series.cumsum() + not_na_series - 1
        )
        return self._from_native_series(result)

    def cum_min(self: Self, *, reverse: bool) -> Self:
        native_series = self._native_series
        result = (
            native_series.cummin(skipna=True)
            if not reverse
            else native_series[::-1].cummin(skipna=True)[::-1]
        )
        return self._from_native_series(result)

    def cum_max(self: Self, *, reverse: bool) -> Self:
        native_series = self._native_series
        result = (
            native_series.cummax(skipna=True)
            if not reverse
            else native_series[::-1].cummax(skipna=True)[::-1]
        )
        return self._from_native_series(result)

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        native_series = self._native_series
        result = (
            native_series.cumprod(skipna=True)
            if not reverse
            else native_series[::-1].cumprod(skipna=True)[::-1]
        )
        return self._from_native_series(result)

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
    ) -> Self:
        result = self._native_series.rolling(
            window=window_size, min_periods=min_periods, center=center
        ).sum()
        return self._from_native_series(result)

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
    ) -> Self:
        result = self._native_series.rolling(
            window=window_size, min_periods=min_periods, center=center
        ).mean()
        return self._from_native_series(result)

    def rolling_var(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
        ddof: int,
    ) -> Self:
        result = self._native_series.rolling(
            window=window_size, min_periods=min_periods, center=center
        ).var(ddof=ddof)
        return self._from_native_series(result)

    def rolling_std(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
        ddof: int,
    ) -> Self:
        result = self._native_series.rolling(
            window=window_size, min_periods=min_periods, center=center
        ).std(ddof=ddof)
        return self._from_native_series(result)

    def __iter__(self: Self) -> Iterator[Any]:
        yield from self._native_series.__iter__()

    def __contains__(self: Self, other: Any) -> bool:
        return (  # type: ignore[no-any-return]
            self._native_series.isna().any()
            if other is None
            else (self._native_series == other).any()
        )

    def is_finite(self: Self) -> Self:
        s = self._native_series
        return self._from_native_series((s > float("-inf")) & (s < float("inf")))

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self:
        pd_method = "first" if method == "ordinal" else method
        native_series = self._native_series
        dtypes = import_dtypes_module(self._version)
        if (
            self._implementation is Implementation.PANDAS
            and self._backend_version < (3,)
            and self.dtype
            in {
                dtypes.Int64,
                dtypes.Int32,
                dtypes.Int16,
                dtypes.Int8,
                dtypes.UInt64,
                dtypes.UInt32,
                dtypes.UInt16,
                dtypes.UInt8,
            }
            and (null_mask := native_series.isna()).any()
        ):
            # crazy workaround for the case of `na_option="keep"` and nullable
            # integer dtypes. This should be supported in pandas > 3.0
            # https://github.com/pandas-dev/pandas/issues/56976
            ranked_series = (
                native_series.to_frame()
                .assign(**{f"{native_series.name}_is_null": null_mask})
                .groupby(f"{native_series.name}_is_null")
                .rank(
                    method=pd_method,
                    na_option="keep",
                    ascending=not descending,
                    pct=False,
                )[native_series.name]
            )

        else:
            ranked_series = native_series.rank(
                method=pd_method,
                na_option="keep",
                ascending=not descending,
                pct=False,
            )

        return self._from_native_series(ranked_series)

    @property
    def str(self: Self) -> PandasLikeSeriesStringNamespace:
        return PandasLikeSeriesStringNamespace(self)

    @property
    def dt(self: Self) -> PandasLikeSeriesDateTimeNamespace:
        return PandasLikeSeriesDateTimeNamespace(self)

    @property
    def cat(self: Self) -> PandasLikeSeriesCatNamespace:
        return PandasLikeSeriesCatNamespace(self)

    @property
    def list(self: Self) -> PandasLikeSeriesListNamespace:
        return PandasLikeSeriesListNamespace(self)

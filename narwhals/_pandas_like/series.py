from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import cast
from typing import overload

import numpy as np

from narwhals._compliant import EagerSeries
from narwhals._pandas_like.series_cat import PandasLikeSeriesCatNamespace
from narwhals._pandas_like.series_dt import PandasLikeSeriesDateTimeNamespace
from narwhals._pandas_like.series_list import PandasLikeSeriesListNamespace
from narwhals._pandas_like.series_str import PandasLikeSeriesStringNamespace
from narwhals._pandas_like.series_struct import PandasLikeSeriesStructNamespace
from narwhals._pandas_like.utils import align_and_extract_native
from narwhals._pandas_like.utils import get_dtype_backend
from narwhals._pandas_like.utils import narwhals_to_native_dtype
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals._pandas_like.utils import object_native_to_narwhals_dtype
from narwhals._pandas_like.utils import rename
from narwhals._pandas_like.utils import select_columns_by_name
from narwhals._pandas_like.utils import set_index
from narwhals.dependencies import is_numpy_array_1d
from narwhals.dependencies import is_numpy_scalar
from narwhals.dependencies import is_pandas_like_series
from narwhals.exceptions import InvalidOperationError
from narwhals.utils import Implementation
from narwhals.utils import import_dtypes_module
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Hashable

    import pandas as pd
    import polars as pl
    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals._arrow.typing import ArrowArray
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray
    from narwhals.typing import _1DArray
    from narwhals.typing import _AnyDArray
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

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


class PandasLikeSeries(EagerSeries[Any]):
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
        # Flag which indicates if, in the final step before applying an operation,
        # the single value behind the PandasLikeSeries should be extract and treated
        # as a Scalar. For example, in `nw.col('a') - nw.lit(3)`, the latter would
        # become a Series of length 1. Rather that doing a full broadcast so it matches
        # the length of the whole dataframe, we just extract the scalar.
        self._broadcast = False

    @property
    def native(self) -> Any:
        return self._native_series

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation.is_pandas_like():
            return self._implementation.to_native_namespace()

        msg = f"Expected pandas/modin/cudf, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def __narwhals_namespace__(self) -> PandasLikeNamespace:
        from narwhals._pandas_like.namespace import PandasLikeNamespace

        return PandasLikeNamespace(
            self._implementation, self._backend_version, self._version
        )

    @overload
    def __getitem__(self: Self, idx: int) -> Any: ...

    @overload
    def __getitem__(self: Self, idx: slice | Sequence[int]) -> Self: ...

    def __getitem__(self: Self, idx: int | slice | Sequence[int]) -> Any | Self:
        if isinstance(idx, int) or is_numpy_scalar(idx):
            return self.native.iloc[idx]
        return self._with_native(self.native.iloc[idx])

    def _with_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self.native,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=version,
        )

    def _with_native(
        self: Self, series: Any, *, preserve_broadcast: bool = False
    ) -> Self:
        result = self.__class__(
            series,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )
        if preserve_broadcast:
            result._broadcast = self._broadcast
        return result

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        context: _FullContext,
        name: str = "",
        dtype: DType | type[DType] | None = None,
        index: Any = None,
    ) -> Self:
        implementation = context._implementation
        backend_version = context._backend_version
        version = context._version
        ns = implementation.to_native_namespace()
        kwds: dict[str, Any] = {}
        if dtype:
            kwds["dtype"] = narwhals_to_native_dtype(
                dtype, None, implementation, backend_version, version
            )
        else:
            if implementation.is_pandas():
                kwds["copy"] = False
            if index is not None and len(index):
                kwds["index"] = index
        return cls.from_native(ns.Series(data, name=name, **kwds), context=context)

    @staticmethod
    def _is_native(obj: Any) -> TypeIs[Any]:
        return is_pandas_like_series(obj)  # pragma: no cover

    @classmethod
    def from_native(cls, data: Any, /, *, context: _FullContext) -> Self:
        return cls(
            data,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _FullContext) -> Self:
        implementation = context._implementation
        arr = data if is_numpy_array_1d(data) else [data]
        native = implementation.to_native_namespace().Series(arr, name="")
        return cls.from_native(native, context=context)

    @property
    def name(self: Self) -> str:
        return self._name

    @property
    def dtype(self: Self) -> DType:
        native_dtype = self.native.dtype
        return (
            native_to_narwhals_dtype(native_dtype, self._version, self._implementation)
            if native_dtype != "object"
            else object_native_to_narwhals_dtype(
                self.native, self._version, self._implementation
            )
        )

    def ewm_mean(
        self: Self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> PandasLikeSeries:
        ser = self.native
        mask_na = ser.isna()
        if self._implementation is Implementation.CUDF:
            if (min_samples == 0 and not ignore_nulls) or (not mask_na.any()):
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
                com, span, half_life, alpha, min_samples, adjust, ignore_na=ignore_nulls
            ).mean()
        result[mask_na] = None
        return self._with_native(result)

    def scatter(self: Self, indices: int | Sequence[int], values: Any) -> Self:
        if isinstance(values, self.__class__):
            values = set_index(
                values.native,
                self.native.index[indices],
                implementation=self._implementation,
                backend_version=self._backend_version,
            )
        s = self.native.copy(deep=True)
        s.iloc[indices] = values
        s.name = self.name
        return self._with_native(s)

    def _scatter_in_place(self: Self, indices: Self, values: Self) -> None:
        # Scatter, modifying original Series. Use with care!
        values_native = set_index(
            values.native,
            self.native.index[indices.native],
            implementation=self._implementation,
            backend_version=self._backend_version,
        )
        if self._implementation is Implementation.PANDAS and parse_version(np) < (2,):
            values_native = values_native.copy()  # pragma: no cover
        min_pd_version = (1, 2)
        if (
            self._implementation is Implementation.PANDAS
            and self._backend_version < min_pd_version
        ):
            self.native.iloc[indices.native.values] = values_native  # noqa: PD011
        else:
            self.native.iloc[indices.native] = values_native

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        pd_dtype = narwhals_to_native_dtype(
            dtype,
            dtype_backend=get_dtype_backend(self.native.dtype, self._implementation),
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )
        return self._with_native(self.native.astype(pd_dtype), preserve_broadcast=True)

    def item(self: Self, index: int | None) -> Any:
        # cuDF doesn't have Series.item().
        if index is None:
            if len(self) != 1:
                msg = (
                    "can only call '.item()' if the Series is of length 1,"
                    f" or an explicit index is provided (Series is of length {len(self)})"
                )
                raise ValueError(msg)
            return self.native.iloc[0]
        return self.native.iloc[index]

    def to_frame(self: Self) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        return PandasLikeDataFrame(
            self.native.to_frame(),
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
            validate_column_names=False,
        )

    def to_list(self: Self) -> list[Any]:
        is_cudf = self._implementation.is_cudf()
        return self.native.to_arrow().to_pylist() if is_cudf else self.native.to_list()

    def is_between(
        self: Self,
        lower_bound: Any,
        upper_bound: Any,
        closed: Literal["left", "right", "none", "both"],
    ) -> PandasLikeSeries:
        ser = self.native
        _, lower_bound = align_and_extract_native(self, lower_bound)
        _, upper_bound = align_and_extract_native(self, upper_bound)
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
        return self._with_native(res).alias(ser.name)

    def is_in(self: Self, other: Any) -> PandasLikeSeries:
        return self._with_native(self.native.isin(other))

    def arg_true(self: Self) -> PandasLikeSeries:
        ser = self.native
        result = ser.__class__(range(len(ser)), name=ser.name, index=ser.index).loc[ser]
        return self._with_native(result)

    def arg_min(self: Self) -> int:
        if self._implementation is Implementation.PANDAS and self._backend_version < (1,):
            return self.native.to_numpy().argmin()
        return self.native.argmin()

    def arg_max(self: Self) -> int:
        ser = self.native
        if self._implementation is Implementation.PANDAS and self._backend_version < (1,):
            return ser.to_numpy().argmax()
        return ser.argmax()

    # Binary comparisons

    def filter(self: Self, predicate: Any) -> PandasLikeSeries:
        if not (
            isinstance(predicate, list) and all(isinstance(x, bool) for x in predicate)
        ):
            _, other_native = align_and_extract_native(self, predicate)
        else:
            other_native = predicate
        return self._with_native(self.native.loc[other_native]).alias(self.name)

    def __eq__(self: Self, other: object) -> PandasLikeSeries:  # type: ignore[override]
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser == other).alias(self.name)

    def __ne__(self: Self, other: object) -> PandasLikeSeries:  # type: ignore[override]
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser != other).alias(self.name)

    def __ge__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser >= other).alias(self.name)

    def __gt__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser > other).alias(self.name)

    def __le__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser <= other).alias(self.name)

    def __lt__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser < other).alias(self.name)

    def __and__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser & other).alias(self.name)

    def __rand__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        ser = cast("pd.Series[Any]", ser)
        return self._with_native(ser.__and__(other)).alias(self.name)

    def __or__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser | other).alias(self.name)

    def __ror__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        ser = cast("pd.Series[Any]", ser)
        return self._with_native(ser.__or__(other)).alias(self.name)

    def __add__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser + other).alias(self.name)

    def __radd__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__radd__(other_native)).alias(self.name)

    def __sub__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser - other).alias(self.name)

    def __rsub__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__rsub__(other_native)).alias(self.name)

    def __mul__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser * other).alias(self.name)

    def __rmul__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__rmul__(other_native)).alias(self.name)

    def __truediv__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser / other).alias(self.name)

    def __rtruediv__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__rtruediv__(other_native)).alias(self.name)

    def __floordiv__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser // other).alias(self.name)

    def __rfloordiv__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__rfloordiv__(other_native)).alias(self.name)

    def __pow__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser**other).alias(self.name)

    def __rpow__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__rpow__(other_native)).alias(self.name)

    def __mod__(self: Self, other: Any) -> PandasLikeSeries:
        ser, other = align_and_extract_native(self, other)
        return self._with_native(ser % other).alias(self.name)

    def __rmod__(self: Self, other: Any) -> PandasLikeSeries:
        _, other_native = align_and_extract_native(self, other)
        return self._with_native(self.native.__rmod__(other_native)).alias(self.name)

    # Unary

    def __invert__(self: PandasLikeSeries) -> PandasLikeSeries:
        return self._with_native(~self.native)

    # Reductions

    def any(self: Self) -> bool:
        return self.native.any()

    def all(self: Self) -> bool:
        return self.native.all()

    def min(self: Self) -> Any:
        return self.native.min()

    def max(self: Self) -> Any:
        return self.native.max()

    def sum(self: Self) -> float:
        return self.native.sum()

    def count(self: Self) -> int:
        return self.native.count()

    def mean(self: Self) -> float:
        return self.native.mean()

    def median(self: Self) -> float:
        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)
        return self.native.median()

    def std(self: Self, *, ddof: int) -> float:
        return self.native.std(ddof=ddof)

    def var(self: Self, *, ddof: int) -> float:
        return self.native.var(ddof=ddof)

    def skew(self: Self) -> float | None:
        ser_not_null = self.native.dropna()
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
        return len(self.native)

    # Transformations

    def is_null(self: Self) -> PandasLikeSeries:
        return self._with_native(self.native.isna(), preserve_broadcast=True)

    def is_nan(self: Self) -> PandasLikeSeries:
        ser = self.native
        if self.dtype.is_numeric():
            return self._with_native(ser != ser, preserve_broadcast=True)  # noqa: PLR0124
        msg = f"`.is_nan` only supported for numeric dtype and not {self.dtype}, did you mean `.is_null`?"
        raise InvalidOperationError(msg)

    def fill_null(
        self: Self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self:
        ser = self.native
        if value is not None:
            _, value = align_and_extract_native(self, value)
            res_ser = self._with_native(ser.fillna(value=value), preserve_broadcast=True)
        else:
            res_ser = self._with_native(
                ser.ffill(limit=limit)
                if strategy == "forward"
                else ser.bfill(limit=limit),
                preserve_broadcast=True,
            )

        return res_ser

    def drop_nulls(self: Self) -> PandasLikeSeries:
        return self._with_native(self.native.dropna())

    def n_unique(self: Self) -> int:
        return self.native.nunique(dropna=False)

    def sample(
        self: Self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self:
        return self._with_native(
            self.native.sample(
                n=n, frac=fraction, replace=with_replacement, random_state=seed
            )
        )

    def abs(self: Self) -> PandasLikeSeries:
        return self._with_native(self.native.abs())

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        result = (
            self.native.cumsum(skipna=True)
            if not reverse
            else self.native[::-1].cumsum(skipna=True)[::-1]
        )
        return self._with_native(result)

    def unique(self: Self, *, maintain_order: bool) -> PandasLikeSeries:
        # pandas always maintains order, as per its docstring:
        # "Uniques are returned in order of appearance"  # noqa: ERA001
        return self._with_native(
            self.native.__class__(self.native.unique(), name=self.name)
        )

    def diff(self: Self) -> PandasLikeSeries:
        return self._with_native(self.native.diff())

    def shift(self: Self, n: int) -> PandasLikeSeries:
        return self._with_native(self.native.shift(n))

    def replace_strict(
        self: Self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: DType | type[DType] | None,
    ) -> PandasLikeSeries:
        tmp_name = f"{self.name}_tmp"
        dtype_backend = get_dtype_backend(self.native.dtype, self._implementation)
        dtype = (
            narwhals_to_native_dtype(
                return_dtype,
                dtype_backend,
                self._implementation,
                self._backend_version,
                self._version,
            )
            if return_dtype
            else None
        )
        namespace = self.__native_namespace__()
        other = namespace.DataFrame(
            {self.name: old, tmp_name: namespace.Series(new, dtype=dtype)}
        )
        result = self._with_native(
            self.native.to_frame().merge(other, on=self.name, how="left")[tmp_name]
        ).alias(self.name)
        if result.is_null().sum() != self.is_null().sum():
            msg = (
                "replace_strict did not replace all non-null values.\n\n"
                f"The following did not get replaced: {self.filter(~self.is_null() & result.is_null()).unique(maintain_order=False).to_list()}"
            )
            raise ValueError(msg)
        return result

    def sort(self: Self, *, descending: bool, nulls_last: bool) -> PandasLikeSeries:
        na_position = "last" if nulls_last else "first"
        return self._with_native(
            self.native.sort_values(ascending=not descending, na_position=na_position)
        ).alias(self.name)

    def alias(self: Self, name: str | Hashable) -> Self:
        if name != self.name:
            return self._with_native(
                rename(
                    self.native,
                    name,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                preserve_broadcast=True,
            )
        return self

    def __array__(self: Self, dtype: Any, *, copy: bool | None) -> _1DArray:
        # pandas used to always return object dtype for nullable dtypes.
        # So, we intercept __array__ and pass to `to_numpy` ourselves to make
        # sure an appropriate numpy dtype is returned.
        return self.to_numpy(dtype=dtype, copy=copy)

    def to_numpy(self: Self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        # the default is meant to be None, but pandas doesn't allow it?
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__array__.html
        copy = copy or self._implementation is Implementation.CUDF
        dtypes = import_dtypes_module(self._version)
        if isinstance(self.dtype, dtypes.Datetime) and self.dtype.time_zone is not None:
            s = self.dt.convert_time_zone("UTC").dt.replace_time_zone(None).native
        else:
            s = self.native

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
                dtype=dtype or PANDAS_TO_NUMPY_DTYPE_NO_MISSING[str(s.dtype)], copy=copy
            )
        return s.to_numpy(dtype=dtype, copy=copy)

    def to_pandas(self: Self) -> pd.Series[Any]:
        if self._implementation is Implementation.PANDAS:
            return self.native
        elif self._implementation is Implementation.CUDF:  # pragma: no cover
            return self.native.to_pandas()
        elif self._implementation is Implementation.MODIN:
            return self.native._to_pandas()
        msg = f"Unknown implementation: {self._implementation}"  # pragma: no cover
        raise AssertionError(msg)

    def to_polars(self: Self) -> pl.Series:
        import polars as pl  # ignore-banned-import

        return pl.from_pandas(self.to_pandas())

    # --- descriptive ---
    def is_unique(self: Self) -> Self:
        return self._with_native(~self.native.duplicated(keep=False)).alias(self.name)

    def null_count(self: Self) -> int:
        return self.native.isna().sum()

    def is_first_distinct(self: Self) -> Self:
        return self._with_native(~self.native.duplicated(keep="first")).alias(self.name)

    def is_last_distinct(self: Self) -> Self:
        return self._with_native(~self.native.duplicated(keep="last")).alias(self.name)

    def is_sorted(self: Self, *, descending: bool) -> bool:
        if not isinstance(descending, bool):
            msg = f"argument 'descending' should be boolean, found {type(descending)}"
            raise TypeError(msg)

        if descending:
            return self.native.is_monotonic_decreasing
        else:
            return self.native.is_monotonic_increasing

    def value_counts(
        self: Self, *, sort: bool, parallel: bool, name: str | None, normalize: bool
    ) -> PandasLikeDataFrame:
        """Parallel is unused, exists for compatibility."""
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        index_name_ = "index" if self._name is None else self._name
        value_name_ = name or ("proportion" if normalize else "count")
        val_count = self.native.value_counts(
            dropna=False, sort=False, normalize=normalize
        ).reset_index()

        val_count.columns = [index_name_, value_name_]

        if sort:
            val_count = val_count.sort_values(value_name_, ascending=False)

        return PandasLikeDataFrame.from_native(val_count, context=self)

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> float:
        return self.native.quantile(q=quantile, interpolation=interpolation)

    def zip_with(self: Self, mask: Any, other: Any) -> PandasLikeSeries:
        ser = self.native
        _, mask = align_and_extract_native(self, mask)
        _, other = align_and_extract_native(self, other)
        res = ser.where(mask, other)
        return self._with_native(res)

    def head(self: Self, n: int) -> Self:
        return self._with_native(self.native.head(n))

    def tail(self: Self, n: int) -> Self:
        return self._with_native(self.native.tail(n))

    def round(self: Self, decimals: int) -> Self:
        return self._with_native(self.native.round(decimals=decimals))

    def to_dummies(
        self: Self, *, separator: str, drop_first: bool
    ) -> PandasLikeDataFrame:
        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        plx = self.__native_namespace__()
        series = self.native
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
        return PandasLikeDataFrame.from_native(result, context=self)

    def gather_every(self: Self, n: int, offset: int) -> Self:
        return self._with_native(self.native.iloc[offset::n])

    def clip(
        self: Self, lower_bound: Self | Any | None, upper_bound: Self | Any | None
    ) -> Self:
        _, lower_bound = (
            align_and_extract_native(self, lower_bound) if lower_bound else (None, None)
        )
        _, upper_bound = (
            align_and_extract_native(self, upper_bound) if upper_bound else (None, None)
        )
        kwargs = {"axis": 0} if self._implementation is Implementation.MODIN else {}
        return self._with_native(self.native.clip(lower_bound, upper_bound, **kwargs))

    def to_arrow(self: Self) -> ArrowArray:
        if self._implementation is Implementation.CUDF:
            return self.native.to_arrow()

        import pyarrow as pa  # ignore-banned-import()

        return pa.Array.from_pandas(self.native)

    def mode(self: Self) -> Self:
        result = self.native.mode()
        result.name = self.name
        return self._with_native(result)

    def cum_count(self: Self, *, reverse: bool) -> Self:
        not_na_series = ~self.native.isna()
        result = (
            not_na_series.cumsum()
            if not reverse
            else len(self) - not_na_series.cumsum() + not_na_series - 1
        )
        return self._with_native(result)

    def cum_min(self: Self, *, reverse: bool) -> Self:
        result = (
            self.native.cummin(skipna=True)
            if not reverse
            else self.native[::-1].cummin(skipna=True)[::-1]
        )
        return self._with_native(result)

    def cum_max(self: Self, *, reverse: bool) -> Self:
        result = (
            self.native.cummax(skipna=True)
            if not reverse
            else self.native[::-1].cummax(skipna=True)[::-1]
        )
        return self._with_native(result)

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        result = (
            self.native.cumprod(skipna=True)
            if not reverse
            else self.native[::-1].cumprod(skipna=True)[::-1]
        )
        return self._with_native(result)

    def rolling_sum(
        self: Self, window_size: int, *, min_samples: int, center: bool
    ) -> Self:
        result = self.native.rolling(
            window=window_size, min_periods=min_samples, center=center
        ).sum()
        return self._with_native(result)

    def rolling_mean(
        self: Self, window_size: int, *, min_samples: int, center: bool
    ) -> Self:
        result = self.native.rolling(
            window=window_size, min_periods=min_samples, center=center
        ).mean()
        return self._with_native(result)

    def rolling_var(
        self: Self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        result = self.native.rolling(
            window=window_size, min_periods=min_samples, center=center
        ).var(ddof=ddof)
        return self._with_native(result)

    def rolling_std(
        self: Self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        result = self.native.rolling(
            window=window_size, min_periods=min_samples, center=center
        ).std(ddof=ddof)
        return self._with_native(result)

    def __iter__(self: Self) -> Iterator[Any]:
        yield from self.native.__iter__()

    def __contains__(self: Self, other: Any) -> bool:
        return self.native.isna().any() if other is None else (self.native == other).any()

    def is_finite(self: Self) -> Self:
        s = self.native
        return self._with_native((s > float("-inf")) & (s < float("inf")))

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self:
        pd_method = "first" if method == "ordinal" else method
        name = self.name
        if (
            self._implementation is Implementation.PANDAS
            and self._backend_version < (3,)
            and self.dtype.is_integer()
            and (null_mask := self.native.isna()).any()
        ):
            # crazy workaround for the case of `na_option="keep"` and nullable
            # integer dtypes. This should be supported in pandas > 3.0
            # https://github.com/pandas-dev/pandas/issues/56976
            ranked_series = (
                self.native.to_frame()
                .assign(**{f"{name}_is_null": null_mask})
                .groupby(f"{name}_is_null")
                .rank(
                    method=pd_method,
                    na_option="keep",
                    ascending=not descending,
                    pct=False,
                )[name]
            )
        else:
            ranked_series = self.native.rank(
                method=pd_method, na_option="keep", ascending=not descending, pct=False
            )
        return self._with_native(ranked_series)

    def hist(
        self: Self,
        bins: list[float | int] | None,
        *,
        bin_count: int | None,
        include_breakpoint: bool,
    ) -> PandasLikeDataFrame:
        from numpy import linspace
        from numpy import zeros

        from narwhals._pandas_like.dataframe import PandasLikeDataFrame

        ns = self.__native_namespace__()
        data: dict[str, Sequence[int | float | str] | _AnyDArray]

        if bin_count == 0 or (bins is not None and len(bins) <= 1):
            data = {}
            if include_breakpoint:
                data["breakpoint"] = []
            data["count"] = []
            return PandasLikeDataFrame.from_native(ns.DataFrame(data), context=self)
        elif self.native.count() < 1:
            if bins is not None:
                data = {"breakpoint": bins[1:], "count": zeros(shape=len(bins) - 1)}
            else:
                count = cast("int", bin_count)
                data = {"breakpoint": linspace(0, 1, count), "count": zeros(shape=count)}
            if not include_breakpoint:
                del data["breakpoint"]
            return PandasLikeDataFrame.from_native(ns.DataFrame(data), context=self)

        elif bin_count is not None:  # use Polars binning behavior
            lower, upper = self.native.min(), self.native.max()
            pad_lowest_bin = False
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            else:
                pad_lowest_bin = True

            bins = linspace(lower, upper, bin_count + 1)
            if pad_lowest_bin and bins is not None:
                bins[0] -= 0.001 * abs(bins[0]) if bins[0] != 0 else 0.001
            bin_count = None

        # pandas (2.2.*) .value_counts(bins=int) adjusts the lowest bin twice, result in improper counts.
        # pandas (2.2.*) .value_counts(bins=[...]) adjusts the lowest bin which should not happen since
        #   the bins were explicitly passed in.
        categories = ns.cut(self.native, bins=bins if bin_count is None else bin_count)
        # modin (0.32.0) .value_counts(...) silently drops bins with empty observations, .reindex
        #   is necessary to restore these bins.
        result = categories.value_counts(dropna=True, sort=False).reindex(
            categories.cat.categories, fill_value=0
        )
        data = {}
        if include_breakpoint:
            data["breakpoint"] = bins[1:] if bins is not None else result.index.right
        data["count"] = result.reset_index(drop=True)
        return PandasLikeDataFrame.from_native(ns.DataFrame(data), context=self)

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
        if not hasattr(self.native, "list"):
            msg = "Series must be of PyArrow List type to support list namespace."
            raise TypeError(msg)
        return PandasLikeSeriesListNamespace(self)

    @property
    def struct(self: Self) -> PandasLikeSeriesStructNamespace:
        if not hasattr(self.native, "struct"):
            msg = "Series must be of PyArrow Struct type to support struct namespace."
            raise TypeError(msg)
        return PandasLikeSeriesStructNamespace(self)

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import TypeVar
from typing import cast
from typing import overload

import polars as pl

from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import ComputeError
from narwhals.exceptions import DuplicateError
from narwhals.exceptions import InvalidOperationError
from narwhals.exceptions import NarwhalsError
from narwhals.exceptions import ShapeError
from narwhals.utils import Version
from narwhals.utils import _DeferredIterable
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals.dtypes import DType
    from narwhals.utils import _StoresNative

    T = TypeVar("T")
    NativeT = TypeVar(
        "NativeT", bound="pl.DataFrame | pl.LazyFrame | pl.Series | pl.Expr"
    )


@overload
def extract_native(obj: _StoresNative[NativeT]) -> NativeT: ...
@overload
def extract_native(obj: T) -> T: ...
def extract_native(obj: _StoresNative[NativeT] | T) -> NativeT | T:
    return obj.native if _is_compliant_polars(obj) else obj


def _is_compliant_polars(
    obj: _StoresNative[NativeT] | Any,
) -> TypeIs[_StoresNative[NativeT]]:
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.series import PolarsSeries

    return isinstance(obj, (PolarsDataFrame, PolarsLazyFrame, PolarsSeries, PolarsExpr))


def extract_args_kwargs(
    args: Iterable[Any], kwds: Mapping[str, Any], /
) -> tuple[Iterator[Any], dict[str, Any]]:
    it_args = (extract_native(arg) for arg in args)
    return it_args, {k: extract_native(v) for k, v in kwds.items()}


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(  # noqa: C901, PLR0912
    dtype: pl.DataType, version: Version, backend_version: tuple[int, ...]
) -> DType:
    dtypes = version.dtypes
    if dtype == pl.Float64:
        return dtypes.Float64()
    if dtype == pl.Float32:
        return dtypes.Float32()
    if hasattr(pl, "Int128") and dtype == pl.Int128:  # pragma: no cover
        # Not available for Polars pre 1.8.0
        return dtypes.Int128()
    if dtype == pl.Int64:
        return dtypes.Int64()
    if dtype == pl.Int32:
        return dtypes.Int32()
    if dtype == pl.Int16:
        return dtypes.Int16()
    if dtype == pl.Int8:
        return dtypes.Int8()
    if hasattr(pl, "UInt128") and dtype == pl.UInt128:  # pragma: no cover
        # Not available for Polars pre 1.8.0
        return dtypes.UInt128()
    if dtype == pl.UInt64:
        return dtypes.UInt64()
    if dtype == pl.UInt32:
        return dtypes.UInt32()
    if dtype == pl.UInt16:
        return dtypes.UInt16()
    if dtype == pl.UInt8:
        return dtypes.UInt8()
    if dtype == pl.String:
        return dtypes.String()
    if dtype == pl.Boolean:
        return dtypes.Boolean()
    if dtype == pl.Object:
        return dtypes.Object()
    if dtype == pl.Categorical:
        return dtypes.Categorical()
    if isinstance_or_issubclass(dtype, pl.Enum):
        if version is Version.V1:
            return dtypes.Enum()  # type: ignore[call-arg]
        categories = _DeferredIterable(
            dtype.categories.to_list
            if backend_version >= (0, 20, 4)
            else lambda: cast("list[str]", dtype.categories)
        )
        return dtypes.Enum(categories)
    if dtype == pl.Date:
        return dtypes.Date()
    if isinstance_or_issubclass(dtype, pl.Datetime):
        return (
            dtypes.Datetime()
            if dtype is pl.Datetime
            else dtypes.Datetime(dtype.time_unit, dtype.time_zone)
        )
    if isinstance_or_issubclass(dtype, pl.Duration):
        return (
            dtypes.Duration()
            if dtype is pl.Duration
            else dtypes.Duration(dtype.time_unit)
        )
    if isinstance_or_issubclass(dtype, pl.Struct):
        fields = [
            dtypes.Field(name, native_to_narwhals_dtype(tp, version, backend_version))
            for name, tp in dtype
        ]
        return dtypes.Struct(fields)
    if isinstance_or_issubclass(dtype, pl.List):
        return dtypes.List(
            native_to_narwhals_dtype(dtype.inner, version, backend_version)
        )
    if isinstance_or_issubclass(dtype, pl.Array):
        outer_shape = dtype.width if backend_version < (0, 20, 30) else dtype.size
        return dtypes.Array(
            native_to_narwhals_dtype(dtype.inner, version, backend_version), outer_shape
        )
    if dtype == pl.Decimal:
        return dtypes.Decimal()
    if dtype == pl.Time:
        return dtypes.Time()
    if dtype == pl.Binary:
        return dtypes.Binary()
    return dtypes.Unknown()


def narwhals_to_native_dtype(  # noqa: C901, PLR0912
    dtype: DType | type[DType], version: Version, backend_version: tuple[int, ...]
) -> pl.DataType:
    dtypes = version.dtypes
    if dtype == dtypes.Float64:
        return pl.Float64()
    if dtype == dtypes.Float32:
        return pl.Float32()
    if dtype == dtypes.Int128 and hasattr(pl, "Int128"):
        # Not available for Polars pre 1.8.0
        return pl.Int128()
    if dtype == dtypes.Int64:
        return pl.Int64()
    if dtype == dtypes.Int32:
        return pl.Int32()
    if dtype == dtypes.Int16:
        return pl.Int16()
    if dtype == dtypes.Int8:
        return pl.Int8()
    if dtype == dtypes.UInt64:
        return pl.UInt64()
    if dtype == dtypes.UInt32:
        return pl.UInt32()
    if dtype == dtypes.UInt16:
        return pl.UInt16()
    if dtype == dtypes.UInt8:
        return pl.UInt8()
    if dtype == dtypes.String:
        return pl.String()
    if dtype == dtypes.Boolean:
        return pl.Boolean()
    if dtype == dtypes.Object:  # pragma: no cover
        return pl.Object()
    if dtype == dtypes.Categorical:
        return pl.Categorical()
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        if version is Version.V1:
            msg = "Converting to Enum is not supported in narwhals.stable.v1"
            raise NotImplementedError(msg)
        if isinstance(dtype, dtypes.Enum):
            return pl.Enum(dtype.categories)
        msg = "Can not cast / initialize Enum without categories present"
        raise ValueError(msg)
    if dtype == dtypes.Date:
        return pl.Date()
    if dtype == dtypes.Time:
        return pl.Time()
    if dtype == dtypes.Binary:
        return pl.Binary()
    if dtype == dtypes.Decimal:
        msg = "Casting to Decimal is not supported yet."
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return pl.Datetime(dtype.time_unit, dtype.time_zone)  # type: ignore[arg-type]
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return pl.Duration(dtype.time_unit)  # type: ignore[arg-type]
    if isinstance_or_issubclass(dtype, dtypes.List):
        return pl.List(narwhals_to_native_dtype(dtype.inner, version, backend_version))
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        fields = [
            pl.Field(
                field.name,
                narwhals_to_native_dtype(field.dtype, version, backend_version),
            )
            for field in dtype.fields
        ]
        return pl.Struct(fields)
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        size = dtype.size
        kwargs = {"width": size} if backend_version < (0, 20, 30) else {"shape": size}
        return pl.Array(
            narwhals_to_native_dtype(dtype.inner, version, backend_version), **kwargs
        )
    return pl.Unknown()  # pragma: no cover


def catch_polars_exception(
    exception: Exception, backend_version: tuple[int, ...]
) -> NarwhalsError | Exception:
    if isinstance(exception, pl.exceptions.ColumnNotFoundError):
        return ColumnNotFoundError(str(exception))
    elif isinstance(exception, pl.exceptions.ShapeError):
        return ShapeError(str(exception))
    elif isinstance(exception, pl.exceptions.InvalidOperationError):
        return InvalidOperationError(str(exception))
    elif isinstance(exception, pl.exceptions.DuplicateError):
        return DuplicateError(str(exception))
    elif isinstance(exception, pl.exceptions.ComputeError):
        return ComputeError(str(exception))
    if backend_version >= (1,) and isinstance(exception, pl.exceptions.PolarsError):
        # Old versions of Polars didn't have PolarsError.
        return NarwhalsError(str(exception))  # pragma: no cover
    elif backend_version < (1,) and "polars.exceptions" in str(
        type(exception)
    ):  # pragma: no cover
        # Last attempt, for old Polars versions.
        return NarwhalsError(str(exception))
    # Just return exception as-is.
    return exception

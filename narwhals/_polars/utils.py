from __future__ import annotations

import abc
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeVar, overload

import polars as pl
import polars.exceptions as pl_exc

import narwhals.exceptions as nw_exc
from narwhals._dispatch import just_dispatch
from narwhals._duration import Interval
from narwhals._utils import (
    Implementation,
    Version,
    _DeferredIterable,
    _StoresCompliant,
    _StoresNative,
    deep_getattr,
    isinstance_or_issubclass,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from typing_extensions import TypeIs

    from narwhals._compliant.typing import Accessor
    from narwhals._polars.dataframe import Method
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.series import PolarsSeries
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType

    T = TypeVar("T")
    NativeT = TypeVar(
        "NativeT", bound="pl.DataFrame | pl.LazyFrame | pl.Series | pl.Expr"
    )

NativeT_co = TypeVar("NativeT_co", "pl.Series", "pl.Expr", covariant=True)
CompliantT_co = TypeVar("CompliantT_co", "PolarsSeries", "PolarsExpr", covariant=True)
CompliantT = TypeVar("CompliantT", "PolarsSeries", "PolarsExpr")

BACKEND_VERSION = Implementation.POLARS._backend_version()
"""Static backend version for `polars`."""

SERIES_ACCEPTS_PD_INDEX: Final[bool] = BACKEND_VERSION >= (0, 20, 7)
"""`pl.Series(values: pd.Index)` fixed in https://github.com/pola-rs/polars/pull/14087"""

SERIES_RESPECTS_DTYPE: Final[bool] = BACKEND_VERSION >= (0, 20, 26)
"""`pl.Series(dtype=...)` fixed in https://github.com/pola-rs/polars/pull/15962

Includes `SERIES_ACCEPTS_PD_INDEX`.
"""

HAS_INT_128 = BACKEND_VERSION >= (1, 18, 0)
"""https://github.com/pola-rs/polars/pull/20232"""

FROM_DICTS_ACCEPTS_MAPPINGS: Final[bool] = BACKEND_VERSION >= (1, 30, 0)
"""`pl.from_dicts(data: Iterable[Mapping[str, Any]])` since https://github.com/pola-rs/polars/pull/22638"""

HAS_UINT_128 = BACKEND_VERSION >= (1, 34, 0)
"""https://github.com/pola-rs/polars/pull/24346"""


@overload
def extract_native(obj: _StoresNative[NativeT]) -> NativeT: ...
@overload
def extract_native(obj: T) -> T: ...
def extract_native(obj: _StoresNative[NativeT] | T) -> NativeT | T:
    return obj.native if _is_compliant_polars(obj) else obj


def _is_compliant_polars(
    obj: _StoresNative[NativeT] | Any,
) -> TypeIs[_StoresNative[NativeT]]:
    from narwhals._polars.dataframe import PolarsDataFrame, PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.series import PolarsSeries

    return isinstance(obj, (PolarsDataFrame, PolarsLazyFrame, PolarsSeries, PolarsExpr))


def extract_args_kwargs(
    args: Iterable[Any], kwds: Mapping[str, Any], /
) -> tuple[Iterator[Any], dict[str, Any]]:
    it_args = (extract_native(arg) for arg in args)
    return it_args, {k: extract_native(v) for k, v in kwds.items()}


@just_dispatch
def _from_native_dtype(dtype: pl.DataType, version: Version) -> DType:
    dtypes = version.dtypes
    return getattr(dtypes, type(dtype).__name__, dtypes.Unknown)()


@_from_native_dtype.register(pl.Enum)
def enum_to_narwhals(dtype: pl.Enum, version: Version) -> DType:
    dtypes = version.dtypes
    if version is Version.V1:
        return dtypes.Enum()  # type: ignore[call-arg]
    return dtypes.Enum(_DeferredIterable(dtype.categories.to_list))


@_from_native_dtype.register(pl.Datetime)
def datetime_to_narwhals(dtype: pl.Datetime, version: Version) -> DType:
    return version.dtypes.Datetime(dtype.time_unit, dtype.time_zone)


@_from_native_dtype.register(pl.Duration)
def duration_to_narwhals(dtype: pl.Duration, version: Version) -> DType:
    return version.dtypes.Duration(dtype.time_unit)


@_from_native_dtype.register(pl.Decimal)
def decimal_to_narwhals(dtype: pl.Decimal, version: Version) -> DType:
    return version.dtypes.Decimal(dtype.precision, dtype.scale)


@_from_native_dtype.register(pl.List)
def list_to_narwhals(dtype: pl.List, version: Version) -> DType:
    return version.dtypes.List(_from_native_dtype(dtype.inner, version))


@_from_native_dtype.register(pl.Array)
def array_to_narwhals(dtype: pl.Array, version: Version) -> DType:
    outer_shape = dtype.width if BACKEND_VERSION < (0, 20, 30) else dtype.size
    return version.dtypes.Array(_from_native_dtype(dtype.inner, version), outer_shape)


@_from_native_dtype.register(pl.Struct)
def struct_to_narwhals(dtype: pl.Struct, version: Version) -> DType:
    Field = dtypes.Field  # noqa: N806
    fields = [Field(name, _from_native_dtype(tp, version)) for name, tp in dtype]
    return version.dtypes.Struct(fields)


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(dtype: pl.DataType, version: Version) -> DType:
    return _from_native_dtype(dtype, version)


dtypes = Version.MAIN.dtypes


def _version_dependent_dtypes() -> dict[type[DType], pl.DataType]:
    if not HAS_INT_128:  # pragma: no cover
        return {}
    nw_to_pl: dict[type[DType], pl.DataType] = {dtypes.Int128: pl.Int128()}
    return nw_to_pl | {dtypes.UInt128: pl.UInt128()} if HAS_UINT_128 else nw_to_pl


NW_TO_PL_DTYPES: Mapping[type[DType], pl.DataType] = {
    dtypes.Float64: pl.Float64(),
    dtypes.Float32: pl.Float32(),
    dtypes.Binary: pl.Binary(),
    dtypes.String: pl.String(),
    dtypes.Boolean: pl.Boolean(),
    dtypes.Categorical: pl.Categorical(),
    dtypes.Date: pl.Date(),
    dtypes.Time: pl.Time(),
    dtypes.Int8: pl.Int8(),
    dtypes.Int16: pl.Int16(),
    dtypes.Int32: pl.Int32(),
    dtypes.Int64: pl.Int64(),
    dtypes.UInt8: pl.UInt8(),
    dtypes.UInt16: pl.UInt16(),
    dtypes.UInt32: pl.UInt32(),
    dtypes.UInt64: pl.UInt64(),
    dtypes.Object: pl.Object(),
    dtypes.Unknown: pl.Unknown(),
    **_version_dependent_dtypes(),
}


def narwhals_to_native_dtype(  # noqa: C901
    dtype: IntoDType, version: Version
) -> pl.DataType:
    dtypes = version.dtypes
    base_type = dtype.base_type()
    if pl_type := NW_TO_PL_DTYPES.get(base_type):
        return pl_type
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        if version is Version.V1:
            msg = "Converting to Enum is not supported in narwhals.stable.v1"
            raise NotImplementedError(msg)
        if isinstance(dtype, dtypes.Enum):
            return pl.Enum(dtype.categories)
        msg = "Can not cast / initialize Enum without categories present"
        raise ValueError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return pl.Datetime(dtype.time_unit, dtype.time_zone)  # type: ignore[arg-type]
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return pl.Duration(dtype.time_unit)  # type: ignore[arg-type]
    if isinstance_or_issubclass(dtype, dtypes.List):
        return pl.List(narwhals_to_native_dtype(dtype.inner, version))
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        fields = [
            pl.Field(field.name, narwhals_to_native_dtype(field.dtype, version))
            for field in dtype.fields
        ]
        return pl.Struct(fields)
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        size = dtype.size
        kwargs = {"width": size} if BACKEND_VERSION < (0, 20, 30) else {"shape": size}
        return pl.Array(narwhals_to_native_dtype(dtype.inner, version), **kwargs)
    if isinstance_or_issubclass(dtype, dtypes.Decimal):
        return pl.Decimal(dtype.precision, dtype.scale)
    return pl.Unknown()  # pragma: no cover


@just_dispatch
def catch_polars_exception(exception: Exception) -> nw_exc.NarwhalsError | Exception:
    # These exceptions are raised when running polars on GPUs via cuDF
    message = str(exception)
    if message.startswith("CUDF failure") or "polars.exceptions" in str(type(exception)):
        return nw_exc.NarwhalsError(message)  # pragma: no cover
    return exception


if BACKEND_VERSION >= (1,):
    exc_types: Iterable[Any] = pl_exc.PolarsError.__subclasses__()
else:
    # Old versions of Polars didn't have PolarsError.
    exc_types = (
        pl_exc.ColumnNotFoundError,
        pl_exc.ShapeError,
        pl_exc.InvalidOperationError,
        pl_exc.DuplicateError,
        pl_exc.ComputeError,
    )


@catch_polars_exception.register(*exc_types)
def _from_native_exception(exception: Exception) -> nw_exc.NarwhalsError:
    tp = getattr(nw_exc, type(exception).__name__, nw_exc.NarwhalsError)
    return tp(str(exception))


class PolarsAnyNamespace(
    _StoresCompliant[CompliantT_co],
    _StoresNative[NativeT_co],
    Protocol[CompliantT_co, NativeT_co],
):
    _accessor: ClassVar[Accessor]

    def __getattr__(self, attr: str) -> Callable[..., CompliantT_co]:
        def func(*args: Any, **kwargs: Any) -> CompliantT_co:
            pos, kwds = extract_args_kwargs(args, kwargs)
            method = deep_getattr(self.native, self._accessor, attr)
            return self.compliant._with_native(method(*pos, **kwds))

        return func


class PolarsDateTimeNamespace(PolarsAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[Accessor] = "dt"

    def truncate(self, every: str) -> CompliantT:
        # Ensure consistent error message is raised.
        Interval.parse(every)
        return self.__getattr__("truncate")(every)

    def offset_by(self, by: str) -> CompliantT:
        # Ensure consistent error message is raised.
        Interval.parse_no_constraints(by)
        return self.__getattr__("offset_by")(by)

    to_string: Method[CompliantT]
    replace_time_zone: Method[CompliantT]
    convert_time_zone: Method[CompliantT]
    timestamp: Method[CompliantT]
    date: Method[CompliantT]
    year: Method[CompliantT]
    month: Method[CompliantT]
    day: Method[CompliantT]
    hour: Method[CompliantT]
    minute: Method[CompliantT]
    second: Method[CompliantT]
    millisecond: Method[CompliantT]
    microsecond: Method[CompliantT]
    nanosecond: Method[CompliantT]
    ordinal_day: Method[CompliantT]
    weekday: Method[CompliantT]
    total_minutes: Method[CompliantT]
    total_seconds: Method[CompliantT]
    total_milliseconds: Method[CompliantT]
    total_microseconds: Method[CompliantT]
    total_nanoseconds: Method[CompliantT]


class PolarsStringNamespace(PolarsAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[Accessor] = "str"

    # NOTE: Use `abstractmethod` if we have defs to implement, but also `Method` usage
    @abc.abstractmethod
    def to_titlecase(self) -> CompliantT: ...

    @abc.abstractmethod
    def zfill(self, width: int) -> CompliantT: ...

    len_chars: Method[CompliantT]
    replace: Method[CompliantT]
    replace_all: Method[CompliantT]
    strip_chars: Method[CompliantT]
    starts_with: Method[CompliantT]
    ends_with: Method[CompliantT]
    contains: Method[CompliantT]
    slice: Method[CompliantT]
    split: Method[CompliantT]
    to_date: Method[CompliantT]
    to_datetime: Method[CompliantT]
    to_lowercase: Method[CompliantT]
    to_uppercase: Method[CompliantT]
    pad_start: Method[CompliantT]
    pad_end: Method[CompliantT]


class PolarsCatNamespace(PolarsAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[Accessor] = "cat"
    get_categories: Method[CompliantT]


class PolarsListNamespace(PolarsAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[Accessor] = "list"

    @abc.abstractmethod
    def len(self) -> CompliantT: ...

    get: Method[CompliantT]

    unique: Method[CompliantT]

    max: Method[CompliantT]

    mean: Method[CompliantT]

    median: Method[CompliantT]

    min: Method[CompliantT]

    sort: Method[CompliantT]

    sum: Method[CompliantT]


class PolarsStructNamespace(PolarsAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[Accessor] = "struct"
    field: Method[CompliantT]

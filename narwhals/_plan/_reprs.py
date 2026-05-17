"""Home of the *make things look nice* bits."""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from narwhals._plan import common
from narwhals.dtypes import (
    Array,
    Binary,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float64,
    FloatType,
    Int64,
    IntegerType,
    List,
    Object,
    String,
    Struct,
    Time,
    Unknown,
)

if TYPE_CHECKING:
    import datetime as dt
    import decimal

    from narwhals.dtypes import DType
    from narwhals.typing import PythonLiteral

__all__ = ("lit_repr",)

LitRepr: TypeAlias = tuple[str | None, str]
"""`(<dtype-repr>, <value-repr>)`

Nested data types may discard `<dtype-repr>`.
"""


@overload
def lit_repr(dtype: DType, value: None, /) -> tuple[str, str]: ...
@overload
def lit_repr(dtype: DType, value: PythonLiteral | None, /) -> tuple[str | None, str]: ...
def lit_repr(dtype: DType, value: PythonLiteral | None, /) -> LitRepr:
    """Return the `dtype` and `value` pieces of the repr for `lit(...)`."""
    return _lit_repr(dtype, value)


@functools.singledispatch
def _lit_repr(dtype: DType, value: PythonLiteral | None, /) -> LitRepr:
    raise NotImplementedError(dtype)


@_lit_repr.register(Object)
@_lit_repr.register(Categorical)
@_lit_repr.register(Enum)
@_lit_repr.register(Unknown)
def _(dtype: Object | Categorical | Enum | Unknown, value: Any | None) -> LitRepr:
    value_s = str(value)
    if isinstance(dtype, Categorical):
        dtype_s = "cat"
    elif isinstance(dtype, Enum):
        dtype_s = "enum"
    else:
        dtype_s = dtype.__class__.__name__.lower()
    if value is not None and isinstance(dtype, (Categorical, Enum)):
        value_s = f"'{value_s}'"
    if value is None and isinstance(dtype, Unknown):
        return None, value_s
    return dtype_s, value_s


@_lit_repr.register(Duration)
def _(dtype: Duration, value: dt.timedelta | None) -> LitRepr:
    dtype_s = "duration"
    if dtype.time_unit != "us":
        dtype_s = f"{dtype_s}[{dtype.time_unit}]"
    args = []
    if value is None:
        return dtype_s, repr(value)
    if value.days:
        args.append(f"{value.days}d")
    if value.seconds:
        args.append(f"{value.seconds}s")
    if value.microseconds:
        args.append(f"{value.microseconds}us")
    if not args:
        args.append("0")
    return dtype_s, f"'{' '.join(args)}'"


@_lit_repr.register(Date)
def _(_dtype: Date, value: dt.date | None) -> LitRepr:
    return "date", (f"'{value}'" if value is not None else repr(value))


@_lit_repr.register(Time)
def _(_dtype: Time, value: dt.time | None) -> LitRepr:
    return "time", (f"'{value}'" if value is not None else repr(value))


@_lit_repr.register(Datetime)
def _(dtype: Datetime, value: dt.datetime | None) -> LitRepr:
    if dtype.time_zone is None and dtype.time_unit == "us":
        args = ""
    elif dtype.time_zone is None:
        args = dtype.time_unit
    else:
        args = f"{dtype.time_unit}, {dtype.time_zone}"
    dtype_s = f"datetime[{args}]" if args else "datetime"
    value_s = f"'{value.isoformat('T')}'" if value else repr(value)
    return dtype_s, value_s


@_lit_repr.register(Decimal)
def _(dtype: Decimal, value: decimal.Decimal | None) -> LitRepr:
    dtype_s = "decimal"
    if not (dtype.precision == 38 and dtype.scale == 0):
        dtype_s = f"{dtype_s}[{dtype.precision},{dtype.scale}]"
    return dtype_s, (f"'{value}'" if value is not None else repr(value))


@_lit_repr.register(Boolean)
@_lit_repr.register(Int64)
@_lit_repr.register(Float64)
@_lit_repr.register(String)
@_lit_repr.register(Binary)
def _(
    dtype: Boolean | Int64 | Float64 | String | Binary, value: float | str | bytes | None
) -> LitRepr:
    # Value repr represents a type from `builtins` unambiguously.
    # Except when we have `None`
    d: str | None
    v = repr(value)
    if value is not None:
        d = None
    elif isinstance(dtype, (Int64, Float64)):
        d = dtype.__class__.__name__[0].lower() + "64"
    elif isinstance(dtype, Boolean):
        d = "bool"
    elif isinstance(dtype, String):
        d = "str"
    else:
        d = "binary"
    return d, v


@_lit_repr.register(FloatType)
@_lit_repr.register(IntegerType)
def _(dtype: FloatType | IntegerType, value: float | None) -> LitRepr:
    # If we have this `DType`, it should be displayed to resolve ambiguity in the value repr
    import string

    tp_name = dtype.__class__.__name__
    code = tp_name[0].lower()
    # we don't have any of these *yet*, but it would break the more naive version in a weird way
    # e.g. `UInt128V2` -> `u1282`
    version_suffix = r"V\d*"
    bits = re.sub(version_suffix, "", tp_name).strip(string.ascii_letters)
    # e.g. `i8`, `f64`, `u128`
    return f"{code}{bits}", repr(value)


@_lit_repr.register(Struct)
def _(dtype: Struct, value: dict[str, Any] | None) -> LitRepr:
    return f"struct[{len(dtype.fields)}]", repr(value)


@_lit_repr.register(List)
def _(dtype: List, value: list[Any] | tuple[Any, ...] | None) -> LitRepr:
    inner, values = _lit_repr_nested_partial(dtype, value)
    dtype_s = f"list[{inner}]" if inner else "list"
    return dtype_s, values


@_lit_repr.register(Array)
def _(dtype: Array, value: list[Any] | tuple[Any, ...] | None) -> LitRepr:
    inner, values = _lit_repr_nested_partial(dtype, value)
    dtype_s = f"{inner}, " if inner else ""
    shape = dtype.shape
    shape_s = repr(shape[0] if len(shape) == 1 else shape)
    dtype_s = f"array[{dtype_s}{shape_s}]"
    return dtype_s, values


# TODO @dangotbanned: Clean up (eventually)
def _lit_repr_nested_partial(
    dtype: Array | List, value: list[Any] | tuple[Any, ...] | None
) -> LitRepr:
    """Messy shared logic for `Array`/`List`.

    - Array deviates by always displaying the dtype, since it also has shape
    """
    if isinstance(dtype, Array):
        leaf: Any = dtype
        for _ in dtype.shape:
            leaf = leaf.inner
        into = leaf
    else:
        into = dtype.inner
    dtype_inner = common.into_dtype(into)
    dtype_always, _ = lit_repr(dtype_inner, None)
    if value is None or not value:
        dtype_s = dtype_always
        values = "None" if value is None else "[]"
    else:
        if isinstance(dtype, Array):
            dtype_s = dtype_always
            _, first = lit_repr(dtype_inner, value[0])
        else:
            dtype_s, first = lit_repr(dtype_inner, value[0])
        if len(value) >= 5:
            values = "..."
            dtype_s = dtype_always
        else:
            it = (lit_repr(dtype_inner, v)[1] for v in value[1:])
            values = ", ".join((first, *it))
        values = f"[{values}]"
    return dtype_s, values

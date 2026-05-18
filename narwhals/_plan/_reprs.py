"""Home of the *make things look nice* bits."""

from __future__ import annotations

import datetime as dt
import functools
import re
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Final, TypeAlias, overload

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
    import decimal
    from collections.abc import Callable

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


def _escape_non_null(value: Any | None, fmt: Callable[[Any], str] = str) -> str:
    return repr(value) if value is None else f"'{fmt(value)}'"


@functools.singledispatch
def _lit_repr(dtype: DType, value: PythonLiteral | None, /) -> LitRepr:
    raise NotImplementedError(dtype)


@_lit_repr.register(Object)
@_lit_repr.register(Unknown)
def _(dtype: Object | Unknown, value: Any | None) -> LitRepr:
    value_s = str(value)
    if value is None and isinstance(dtype, Unknown):
        return None, value_s
    return dtype.__class__.__name__.lower(), value_s


@_lit_repr.register(Categorical)
@_lit_repr.register(Enum)
def _(dtype: Categorical | Enum, value: Any | None) -> LitRepr:
    return "enum" if isinstance(dtype, Enum) else "cat", _escape_non_null(value)


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
    return "date", _escape_non_null(value)


@_lit_repr.register(Time)
def _(_dtype: Time, value: dt.time | None) -> LitRepr:
    return "time", _escape_non_null(value)


@_lit_repr.register(Datetime)
def _(dtype: Datetime, value: dt.datetime | None) -> LitRepr:
    if dtype.time_zone is None and dtype.time_unit == "us":
        args = ""
    elif dtype.time_zone is None:
        args = dtype.time_unit
    else:
        args = f"{dtype.time_unit}, {dtype.time_zone}"
    dtype_s = f"datetime[{args}]" if args else "datetime"
    return dtype_s, _escape_non_null(value, dt.datetime.isoformat)


@_lit_repr.register(Decimal)
def _(dtype: Decimal, value: decimal.Decimal | None) -> LitRepr:
    dtype_s = "decimal"
    if not (dtype.precision == 38 and dtype.scale == 0):
        dtype_s = f"{dtype_s}[{dtype.precision},{dtype.scale}]"
    return dtype_s, _escape_non_null(value)


@_lit_repr.register(Boolean)
@_lit_repr.register(Int64)
@_lit_repr.register(Float64)
@_lit_repr.register(String)
@_lit_repr.register(Binary)
def _(
    dtype: Boolean | Int64 | Float64 | String | Binary, value: float | str | bytes | None
) -> LitRepr:
    if value is not None:
        return None, repr(value)
    if isinstance(dtype, (Int64, Float64)):
        return _numeric(dtype, value)
    if isinstance(dtype, Boolean):
        d = "bool"
    elif isinstance(dtype, String):
        d = "str"
    else:
        d = "binary"
    return d, repr(value)


_strip_version_suffix: Final = functools.partial(re.compile(r"V\d*").sub, "")
"""Remove any trailing `"V1"`, `"V2"`, etc from the name of a `DType` class."""


@_lit_repr.register(FloatType)
@_lit_repr.register(IntegerType)
def _numeric(dtype: FloatType | IntegerType, value: float | None) -> LitRepr:
    # Always display to resolve ambiguity in the value repr (e.g. `i8`, `f32`, `u128`)
    tp_name = dtype.__class__.__name__
    char = tp_name[0].lower()
    return (f"{char}{_strip_version_suffix(tp_name).strip(ascii_letters)}", repr(value))


@_lit_repr.register(Struct)
def _(dtype: Struct, value: dict[str, Any] | None) -> LitRepr:
    return f"struct[{len(dtype.fields)}]", repr(value)


@_lit_repr.register(List)
def _(dtype: List, value: list[Any] | tuple[Any, ...] | None) -> LitRepr:
    inner, values = _lit_repr_nested_partial(dtype, value)
    return f"list[{inner}]" if inner else "list", values


@_lit_repr.register(Array)
def _(dtype: Array, value: list[Any] | tuple[Any, ...] | None) -> LitRepr:
    inner, values = _lit_repr_nested_partial(dtype, value)
    dtype_s = f"{inner}, " if inner else ""
    shape = dtype.shape
    shape_s = repr(shape[0] if len(shape) == 1 else shape)
    return f"array[{dtype_s}{shape_s}]", values


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

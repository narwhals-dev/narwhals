# mypy: disable-error-code="misc", warn-unused-configs=True
# # NOTE: https://discuss.python.org/t/make-replace-stop-interfering-with-variance-inference/96092
from __future__ import annotations

import datetime as dt
import decimal
import functools
import re
from functools import cache
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, final

from narwhals._plan import common
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import Constructor
from narwhals._plan._guards import is_python_literal_type
from narwhals._plan.exceptions import literal_type_error
from narwhals._plan.typing import (
    NativeSeriesT,
    NativeSeriesT_co,
    PythonLiteralT,
    PythonLiteralT_co,
)
from narwhals._utils import Version
from narwhals.dtypes import (
    Array,
    Binary,
    Boolean,
    Decimal,
    Enum,
    Field,
    Float64,
    FloatType,
    Int64,
    IntegerType,
    List,
    Object,
    String,
    Struct,
    Unknown,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.series import Series
    from narwhals.dtypes import Categorical, Date, Datetime, DType, Duration, Time
    from narwhals.typing import IntoDType, NonNestedDType, PythonLiteral

__all__ = ("Lit", "LitSeries", "lit", "lit_series")

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype


# TODO @dangotbanned: (low-prio) Define `__str__` to use (`value`, `dtype`)-order instead
# - Generated from alphabetical-ordered slots
# - Will break some doctests on a change
@final
class Lit(Constructor, Generic[PythonLiteralT_co], dtype=get_dtype(), dispatch="Scalar"):
    """An expression representing a scalar literal value.

    >>> import narwhals._plan as nw
    >>> expr = nw.lit(1)
    >>> expr._ir
    lit(1)
    >>> expr.meta.is_literal()
    True
    >>> expr._ir.is_scalar()
    True
    """

    __slots__ = ("dtype", "value")
    value: PythonLiteralT_co
    """The literal value."""
    dtype: DType
    """The data type inferred or explicitly given at construction time."""

    @property
    def name(self) -> str:
        return "literal"

    def is_scalar(self) -> bool:
        return True

    def changes_length(self) -> bool:
        return False

    is_length_preserving = changes_length

    @staticmethod
    def from_python(
        value: PythonLiteralT,
        dtype: IntoDType | None = None,
        version: Version = Version.MAIN,
    ) -> Lit[PythonLiteralT]:
        dtype = (
            _py_value_to_dtype(value, version, allow_null=True)
            if dtype is None
            else common.into_dtype(dtype)
        )
        return Lit(value=value, dtype=dtype)

    def __repr__(self) -> str:
        dtype, value = lit_repr(self.dtype, self.value)
        ctor = f"lit[{dtype}]" if dtype else "lit"
        return f"{ctor}({value})"

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        dtype = self.dtype
        value: Any = self.value
        yield from (id(value) if dtype.is_nested() else value, dtype)


@final
class LitSeries(
    Constructor, Generic[NativeSeriesT_co], dtype=get_dtype(), dispatch="Expr"
):
    """An expression representing a series literal.

    >>> import narwhals._plan as nw
    >>> series = nw.Series.from_iterable([1], backend="pyarrow")
    >>> expr = nw.lit(series)
    >>> expr._ir
    lit(Series[pa.ChunkedArray])
    >>> expr.meta.is_literal()
    True

    A length-1 series literal is not considered scalar
    >>> expr._ir.is_scalar()
    False
    """

    __slots__ = ("dtype", "value")
    value: Series[NativeSeriesT_co]
    dtype: DType

    def is_scalar(self) -> bool:
        return False

    is_length_preserving = changes_length = is_scalar

    @staticmethod
    def from_series(series: Series[NativeSeriesT], /) -> LitSeries[NativeSeriesT]:
        return LitSeries(value=series, dtype=series.dtype)

    @property
    def name(self) -> str:
        return self.value.name

    @property
    def native(self) -> NativeSeriesT_co:
        return self.value.to_native()

    @property
    def version(self) -> Version:  # pragma: no cover
        return self.value.version

    def __repr__(self) -> str:
        return f"lit(Series[{self.value._compliant.__narwhals_repr_name__()}])"

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Adding `Series.__eq__` means this needed a manual override
        yield from (self.name, self.dtype, id(self.value))


lit = Lit.from_python
lit_series = LitSeries.from_series


def _py_value_to_dtype(
    obj: PythonLiteral, version: Version = Version.MAIN, *, allow_null: bool
) -> DType:
    # NOTE: Surely mypy must have fixed `_lru_cache_wrapper` hashable in a new version?
    if dtype := _py_type_to_dtype(type(obj), version):  # type: ignore[arg-type]
        if allow_null or not isinstance(dtype, Unknown):
            return dtype
        msg = "Nested dtypes containing nulls are not yet supported"
        raise TypeError(msg)
    if not isinstance(obj, (list, dict, tuple)):
        # Just a type narrowing issue
        msg = f"Expected unreachable, got {obj!r}"
        raise NotImplementedError(msg)

    if not obj:
        msg = "Cannot infer dtype for empty nested structure. Please provide an explicit dtype parameter."
        raise TypeError(msg)
    if not isinstance(obj, dict):
        first_value = next((el for el in obj if el is not None), None)
        return List(_py_value_to_dtype(first_value, version, allow_null=False))
    return Struct(
        [
            Field(k, _py_value_to_dtype(v, version, allow_null=False))
            for k, v in obj.items()
        ]
    )


@cache
def _py_type_to_dtype(
    py_type: type[PythonLiteral], version: Version = Version.MAIN, /
) -> NonNestedDType | None:
    """SAFETY.

    Cache size is bound by these dimensions:

        n_valid_py_types = len(non_nested) + len([list, dict, tuple])
        maxsize = n_valid_py_types * len(Version)
    """
    dtypes = version.dtypes
    non_nested: dict[type[PythonLiteral], type[NonNestedDType]] = {
        int: dtypes.Int64,
        float: dtypes.Float64,
        str: dtypes.String,
        bool: dtypes.Boolean,
        dt.datetime: dtypes.Datetime,
        dt.date: dtypes.Date,
        dt.time: dtypes.Time,
        dt.timedelta: dtypes.Duration,
        bytes: dtypes.Binary,
        decimal.Decimal: dtypes.Decimal,
        type(None): dtypes.Unknown,
    }
    if dtype := non_nested.get(py_type):
        return dtype()
    if not is_python_literal_type(py_type):
        raise literal_type_error(py_type)
    return None


_dtypes = Version.MAIN.dtypes


LitRepr: TypeAlias = tuple[str | None, str]
"""`(<dtype-repr>, <value-repr>)`

Nested data types may discard `<dtype-repr>`.
"""


# TODO @dangotbanned: Need special typing for passing `None` triggering an overload that ensures `tuple[str, str]`
@functools.singledispatch
def lit_repr(dtype: DType, value: PythonLiteral | None, /) -> LitRepr:
    raise NotImplementedError(dtype)


@lit_repr.register(_dtypes.Object)
@lit_repr.register(_dtypes.Categorical)
@lit_repr.register(_dtypes.Enum)
@lit_repr.register(_dtypes.Unknown)
def _(dtype: Object | Categorical | Enum | Unknown, value: Any | None) -> LitRepr:
    value_s = str(value)
    if isinstance(dtype, _dtypes.Categorical):
        dtype_s = "cat"
    elif isinstance(dtype, _dtypes.Enum):
        dtype_s = "enum"
    else:
        dtype_s = dtype.__class__.__name__.lower()
    if value is not None and isinstance(dtype, (_dtypes.Categorical, _dtypes.Enum)):
        value_s = f"'{value_s}'"
    if value is None and isinstance(dtype, _dtypes.Unknown):
        return None, value_s
    return dtype_s, value_s


@lit_repr.register(_dtypes.Duration)
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


@lit_repr.register(_dtypes.Date)
def _(_dtype: Date, value: dt.date | None) -> LitRepr:
    return "date", (f"'{value}'" if value is not None else repr(value))


@lit_repr.register(_dtypes.Time)
def _(_dtype: Time, value: dt.time | None) -> LitRepr:
    return "time", (f"'{value}'" if value is not None else repr(value))


@lit_repr.register(_dtypes.Datetime)
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


@lit_repr.register(Decimal)
def _(dtype: Decimal, value: decimal.Decimal | None) -> LitRepr:
    dtype_s = "decimal"
    if not (dtype.precision == 38 and dtype.scale == 0):
        dtype_s = f"{dtype_s}[{dtype.precision},{dtype.scale}]"
    return dtype_s, (f"'{value}'" if value is not None else repr(value))


@lit_repr.register(_dtypes.Boolean)
@lit_repr.register(_dtypes.Int64)
@lit_repr.register(_dtypes.Float64)
@lit_repr.register(_dtypes.String)
@lit_repr.register(_dtypes.Binary)
def _(
    dtype: Boolean | Int64 | Float64 | String | Binary,
    value: bool | float | str | bytes | None,  # noqa: FBT001
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


@lit_repr.register(FloatType)
@lit_repr.register(IntegerType)
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


@lit_repr.register(Struct)
def _(dtype: Struct, value: dict[str, Any] | None) -> LitRepr:
    return f"struct[{len(dtype.fields)}]", repr(value)


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


@lit_repr.register(List)
def _(dtype: List, value: list[Any] | tuple[Any, ...] | None) -> LitRepr:
    inner, values = _lit_repr_nested_partial(dtype, value)
    dtype_s = f"list[{inner}]" if inner else "list"
    return dtype_s, values



@lit_repr.register(Array)
def _(dtype: Array, value: list[Any] | tuple[Any, ...] | None) -> LitRepr:
    inner, values = _lit_repr_nested_partial(dtype, value)
    dtype_s = f"{inner}, " if inner else ""
    shape = dtype.shape
    shape_s = repr(shape[0] if len(shape) == 1 else shape)
    dtype_s = f"array[{dtype_s}{shape_s}]"
    return dtype_s, values

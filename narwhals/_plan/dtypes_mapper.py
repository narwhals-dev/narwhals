"""Implements a subset of [`aexpr::function_expr::schema::FieldsMapper`].

[`aexpr::function_expr::schema::FieldsMapper`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L468-L830
"""

from __future__ import annotations

from functools import cache, lru_cache
from typing import TYPE_CHECKING, Protocol

from narwhals._plan.exceptions import invalid_dtype_operation_error
from narwhals._utils import Version
from narwhals.dtypes import (
    Array,
    Binary,
    Boolean,
    Decimal,
    DType,
    Duration,
    Float32,
    Float64,
    FloatType,
    Int8,
    Int16,
    IntegerType,
    List,
    NestedType,
    NumericType,
    String,
    Struct as Struct,  # noqa: PLC0414
    TemporalType,
    UInt8,
    UInt16,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.expressions import ExprIR
    from narwhals._plan.schema import FrozenSchema
    from narwhals.typing import TimeUnit

    class _HasChildExpr(Protocol):
        @property
        def expr(self) -> ExprIR: ...


dtypes = Version.MAIN.dtypes
dtypes_v1 = Version.V1.dtypes

_HAS_INNER = List, Array
_INVALID_VAR = dtypes.Duration
_INVALID_SUM = String, Binary, List, Array, Struct
_PRIMITIVE_NUMERIC = IntegerType, FloatType

I128 = dtypes.Int128()
I64 = dtypes.Int64()
I32 = dtypes.Int32()
I16 = dtypes.Int16()
I8 = dtypes.Int8()
U64 = dtypes.UInt64()
U32 = dtypes.UInt32()
F64 = dtypes.Float64()
F32 = dtypes.Float32()

IDX_DTYPE = I64
"""TODO @dangotbanned: Unify `IDX_DTYPE` as backends are mixed:

- UInt32 ([polars] excluding `bigidx`)
- UInt64 ([pyarrow] in some cases)
- Int64 (most backends)

[polars]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-core/src/datatypes/aliases.rs#L14
[pyarrow]: https://github.com/narwhals-dev/narwhals/blob/bbc5d4492667eb3b9a364caba35e51308c86cf7d/narwhals/_arrow/dataframe.py#L534-L547
"""

BOOLEAN_DTYPE = dtypes.Boolean()
STRING_DTYPE = String()
DATE_DTYPE = dtypes.Date()


# TODO @dangotbanned: Make this an `ExprIR.__init_subclass__` option?
def resolve_dtype_root(expr: _HasChildExpr, schema: FrozenSchema, /) -> DType:
    """Call `expr.expr._resolve_dtype(schema)`."""
    return expr.expr._resolve_dtype(schema)


def _inner_into_dtype(dtype: List | Array, /) -> DType:
    """Return the initialized inner dtype.

    `dtype.inner: IntoDType` -> `DType`
    """
    return dtype.inner if not isinstance(dtype.inner, type) else dtype.inner()


def map_dtype(
    mapper: Callable[[DType], DType], /, dtype: DType, *, map_inner: bool
) -> DType:
    if map_inner and isinstance(dtype, _HAS_INNER):
        inner_in = _inner_into_dtype(dtype)
        inner_out = mapper(inner_in)
        if inner_out == inner_in:
            return dtype
        if isinstance(dtype, List):
            return List(inner_out)
        return Array(inner_out, dtype.shape)
    return mapper(dtype)


def float_dtype(dtype: DType) -> Float32 | Float64:
    return F32 if type(dtype) is Float32 else F64


def numeric_to_float_dtype_coerce_decimal(dtype: DType) -> DType:
    # `coerce_decimal: false` is only used for an expression we don't support
    # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L271-L272
    if isinstance(dtype, (NumericType, Boolean)):
        return float_dtype(dtype)
    return dtype


def inner_dtype(
    dtype: DType,
    method_name: str = "",
    expected: tuple[type[List | Array], ...] = (List,),
) -> DType:
    """Validate we have `expected` nested type, then unwrap `dtype.inner`."""
    if isinstance(dtype, expected):
        return _inner_into_dtype(dtype)  # type: ignore[arg-type]
    if method_name:
        raise invalid_dtype_operation_error(dtype, method_name, *expected)
    msg = f"expected {' or '.join(str(tp) for tp in expected)} type, got dtype: {dtype}"
    raise InvalidOperationError(msg)


def moment_dtype(dtype: DType) -> DType:
    return map_dtype(_moment_dtype, dtype, map_inner=True)


def var_dtype(dtype: DType) -> DType:
    return map_dtype(_var_dtype, dtype, map_inner=True)


def sum_dtype(dtype: DType) -> DType:
    if isinstance(dtype, _INVALID_SUM):
        raise invalid_dtype_operation_error(dtype, "sum")
    return _sum_transform().get(type(dtype), dtype)


# NOTE @dangotbanned: If we add `arr.sum`, expose `inner_dtype` parameters
def nested_sum_dtype(dtype: DType) -> DType:
    return sum_dtype(inner_dtype(dtype, "list.sum"))


def nested_mean_median_dtype(dtype: DType) -> DType:
    inner = inner_dtype(dtype, expected=(List,))
    if inner.is_temporal():
        return _date_to_datetime_transform().get(type(inner), inner)
    return float_dtype(inner)


def list_join_dtype(dtype: DType) -> String:
    inner = inner_dtype(dtype, "list.join")
    if isinstance(inner, String):
        return inner
    raise invalid_dtype_operation_error(dtype, "list.join", List(STRING_DTYPE))


def _var_dtype(dtype: DType) -> DType:
    if isinstance(dtype, _INVALID_VAR):
        raise invalid_dtype_operation_error(dtype, "var")
    return _moment_dtype(dtype)


def _moment_dtype(dtype: DType) -> DType:
    return _moment_transform().get(type(dtype), dtype)


@cache
def _date_to_datetime_transform() -> Mapping[type[DType], DType]:
    return {dtypes.Date: dtypes.Datetime(), dtypes_v1.Date: dtypes_v1.Datetime()}


@cache
def _moment_transform() -> Mapping[type[DType], DType]:
    return {dtypes.Boolean: F64, dtypes.Decimal: F64, **_date_to_datetime_transform()}


@cache
def _sum_transform() -> Mapping[type[DType], DType]:
    return {
        dtypes.Int8: I64,
        dtypes.UInt8: I64,
        dtypes.Int16: I64,
        dtypes.UInt16: I64,
        dtypes.Boolean: IDX_DTYPE,
    }


@cache
def _diff_int_transform() -> Mapping[type[DType], DType]:
    return {
        dtypes.UInt64: I64,
        dtypes.UInt32: I64,
        dtypes.UInt16: I32,
        dtypes.UInt8: dtypes.Int16(),
    }


@cache
def _duration_type() -> Mapping[type[DType], type[Duration]]:
    return {
        dtypes.Datetime: dtypes.Duration,
        dtypes_v1.Datetime: dtypes_v1.Duration,
        dtypes.Date: dtypes.Duration,
        dtypes_v1.Date: dtypes_v1.Duration,
        dtypes.Time: dtypes.Duration,
    }


def diff_dtype(dtype: DType) -> DType:
    # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L246-L261
    if dtype.is_temporal():
        if duration := _duration_type().get(type(dtype)):
            tu: TimeUnit = "us"
            if isinstance(dtype, dtypes.Datetime):
                tu = dtype.time_unit
            if isinstance(dtype, dtypes.Time):
                tu = "ns"
            return duration(tu)
        return dtype
    if int_dtype := _diff_int_transform().get(type(dtype)):
        return int_dtype
    if isinstance(dtype, dtypes.Decimal):
        return dtypes.Decimal(38, dtype.scale)
    return dtype


_TrueDivSpecial: TypeAlias = "UInt8 | Int8 | UInt16 | Int16 | Boolean | Float32 | Decimal"
"""Data types with complex special-casing in `__truediv__`.

They are split out to reduce the number of narrowing checks all other cases need to make.
E.g., the sooner we can rule them out on the *left*, the faster we can handle more common pairs.
"""


@lru_cache(maxsize=8)
def truediv_dtype(left: DType, right: DType, /) -> DType:
    """Adapted from [`schema::get_truediv_dtype`].

    [`schema::get_truediv_dtype`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L749-L877
    """
    if isinstance(left, (NestedType, String)) or isinstance(right, (NestedType, String)):
        if String in {left.base_type(), right.base_type()}:
            raise _invalid_division_error(String)
        # NOTE: Low priority + high complexity
        msg = f"Division with nested data types is not yet implemented: \n(left: {left}, right: {right})"
        raise NotImplementedError(msg)

    if isinstance(left, TemporalType):
        if isinstance(left, Duration):
            if left.base_type() is right.base_type():
                return F64
            if isinstance(right, _PRIMITIVE_NUMERIC):
                return left
        raise _invalid_division_error(left)

    if _is_truediv_special(left) and (dtype := _truediv_special(left, right)):
        return dtype

    if isinstance(left, _PRIMITIVE_NUMERIC):
        return F64

    return left


def _is_truediv_special(left: DType, /) -> TypeIs[_TrueDivSpecial]:
    return isinstance(left, (UInt8, Int8, UInt16, Int16, Boolean, Float32, Decimal))


if not TYPE_CHECKING:
    # NOTE: Hack to get mypy to recognize a cached `TypeIs`
    _is_truediv_special = lru_cache(maxsize=16)(_is_truediv_special)


def _truediv_special(left: _TrueDivSpecial, right: DType, /) -> DType | None:
    ret: DType | None = None

    if not isinstance(left, NumericType):
        if isinstance(right, Float32):
            ret = right
        elif isinstance(right, (NumericType, Boolean)):
            ret = F64

    elif isinstance(left, Decimal):
        if isinstance(right, Decimal):
            ret = dtypes.Decimal(38, max(left.scale, right.scale))

    elif isinstance(left, Float32):
        if isinstance(right, (UInt8, Int8, UInt16, Int16)):
            ret = F32
        elif isinstance(right, (IntegerType, Float64)):
            ret = F64
        else:
            ret = F32

    elif isinstance(right, Float32) and isinstance(left, (UInt8, Int8, UInt16, Int16)):
        ret = F32

    return ret


def _invalid_division_error(dtype: DType | type[DType], /) -> InvalidOperationError:
    msg = f"Division with {dtype.base_type()!r} datatypes is not allowed"
    return InvalidOperationError(msg)

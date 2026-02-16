"""Implements a subset of [`aexpr::function_expr::schema::FieldsMapper`].

[`aexpr::function_expr::schema::FieldsMapper`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L468-L830
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from narwhals._plan.exceptions import invalid_dtype_operation_error
from narwhals._utils import Version
from narwhals.dtypes import (
    Array,
    Binary,
    Boolean,
    Duration,
    Float32,
    Float64,
    List,
    NumericType,
    String,
    Struct,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit

dtypes = Version.MAIN.dtypes
dtypes_v1 = Version.V1.dtypes

_HAS_INNER = List, Array
_INVALID_VAR = dtypes.Duration
_INVALID_SUM = String, Binary, List, Array, Struct

I64 = dtypes.Int64()
U32 = dtypes.UInt32()
F32 = dtypes.Float32()
F64 = dtypes.Float64()

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
        dtypes.UInt16: dtypes.Int32(),
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

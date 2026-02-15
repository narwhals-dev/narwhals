"""Implements a subset of [`aexpr::function_expr::schema::FieldsMapper`].

[`aexpr::function_expr::schema::FieldsMapper`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L468-L830
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from narwhals._plan.exceptions import invalid_dtype_operation_error
from narwhals._utils import Version
from narwhals.dtypes import Array, Binary, List, String, Struct

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from narwhals.dtypes import DType

dtypes = Version.MAIN.dtypes
dtypes_v1 = Version.V1.dtypes

_HAS_INNER = List, Array
_INVALID_VAR = dtypes.Duration
_INVALID_SUM = String, Binary, List, Array, Struct

I64 = dtypes.Int64()
U32 = dtypes.UInt32()

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


def map_dtype(
    mapper: Callable[[DType], DType], /, dtype: DType, *, map_inner: bool
) -> DType:
    if map_inner and isinstance(dtype, _HAS_INNER):
        inner_in = dtype.inner if not isinstance(dtype.inner, type) else dtype.inner()
        inner_out = mapper(inner_in)
        if inner_out == inner_in:
            return dtype
        if isinstance(dtype, List):
            return List(inner_out)
        return Array(inner_out, dtype.shape)
    return mapper(dtype)


def moment_dtype(dtype: DType) -> DType:
    return map_dtype(_moment_dtype, dtype, map_inner=True)


def var_dtype(dtype: DType) -> DType:
    return map_dtype(_var_dtype, dtype, map_inner=True)


def sum_dtype(dtype: DType) -> DType:
    if isinstance(dtype, _INVALID_SUM):
        raise invalid_dtype_operation_error(dtype, "sum")
    return _sum_transform().get(type(dtype), dtype)


def _var_dtype(dtype: DType) -> DType:
    if isinstance(dtype, _INVALID_VAR):
        raise invalid_dtype_operation_error(dtype, "var")
    return _moment_dtype(dtype)


def _moment_dtype(dtype: DType) -> DType:
    return _moment_transform().get(type(dtype), dtype)


@cache
def _moment_transform() -> Mapping[type[DType], DType]:
    return {
        dtypes.Boolean: dtypes.Float64(),
        dtypes.Date: dtypes.Datetime(),
        dtypes_v1.Date: dtypes_v1.Datetime(),
        dtypes.Decimal: dtypes.Float64(),
    }


@cache
def _sum_transform() -> Mapping[type[DType], DType]:
    return {
        dtypes.Int8: I64,
        dtypes.UInt8: I64,
        dtypes.Int16: I64,
        dtypes.UInt16: I64,
        dtypes.Boolean: IDX_DTYPE,
    }

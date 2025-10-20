"""Native functions, aliased and/or with behavior aligned to `polars`."""

from __future__ import annotations

import typing as t
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import (
    cast_for_truediv,
    chunked_array as _chunked_array,
    floordiv_compat as floordiv,
)
from narwhals._plan import expressions as ir
from narwhals._plan.arrow import options
from narwhals._plan.expressions import functions as F, operators as ops
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._arrow.typing import Incomplete, PromoteOptions
    from narwhals._plan.arrow.series import ArrowSeries
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        ArrowAny,
        BinaryComp,
        BinaryLogical,
        BinaryNumericTemporal,
        BinOp,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrArrayAny,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        DataType,
        DataTypeRemap,
        DataTypeT,
        IntegerScalar,
        IntegerType,
        LargeStringType,
        NativeScalar,
        Scalar,
        ScalarAny,
        ScalarT,
        StringScalar,
        StringType,
        UnaryFunction,
    )
    from narwhals.typing import ClosedInterval, IntoArrowSchema

BACKEND_VERSION = Implementation.PYARROW._backend_version()

IntoColumnAgg: TypeAlias = Callable[[str], ir.AggExpr]
"""Helper constructor for single-column aggregations."""

is_null = pc.is_null
is_not_null = t.cast("UnaryFunction[ScalarAny,pa.BooleanScalar]", pc.is_valid)
is_nan = pc.is_nan
is_finite = pc.is_finite

and_ = t.cast("BinaryLogical", pc.and_kleene)
or_ = t.cast("BinaryLogical", pc.or_kleene)
xor = t.cast("BinaryLogical", pc.xor)

eq = t.cast("BinaryComp", pc.equal)
not_eq = t.cast("BinaryComp", pc.not_equal)
gt_eq = t.cast("BinaryComp", pc.greater_equal)
gt = t.cast("BinaryComp", pc.greater)
lt_eq = t.cast("BinaryComp", pc.less_equal)
lt = t.cast("BinaryComp", pc.less)


add = t.cast("BinaryNumericTemporal", pc.add)
sub = pc.subtract
multiply = pc.multiply


def truediv(lhs: Any, rhs: Any) -> Any:
    return pc.divide(*cast_for_truediv(lhs, rhs))


def modulus(lhs: Any, rhs: Any) -> Any:
    floor_div = floordiv(lhs, rhs)
    return sub(lhs, multiply(floor_div, rhs))


_DISPATCH_BINARY: Mapping[type[ops.Operator], BinOp] = {
    ops.Eq: eq,
    ops.NotEq: not_eq,
    ops.Lt: lt,
    ops.LtEq: lt_eq,
    ops.Gt: gt,
    ops.GtEq: gt_eq,
    ops.Add: add,
    ops.Sub: sub,
    ops.Multiply: multiply,
    ops.TrueDivide: truediv,
    ops.FloorDivide: floordiv,
    ops.Modulus: modulus,
    ops.And: and_,
    ops.Or: or_,
    ops.ExclusiveOr: xor,
}

_IS_BETWEEN: Mapping[ClosedInterval, tuple[BinaryComp, BinaryComp]] = {
    "left": (gt_eq, lt),
    "right": (gt, lt_eq),
    "none": (gt, lt),
    "both": (gt_eq, lt_eq),
}
IS_FIRST_LAST_DISTINCT: Mapping[type[ir.boolean.BooleanFunction], IntoColumnAgg] = {
    ir.boolean.IsFirstDistinct: ir.min,
    ir.boolean.IsLastDistinct: ir.max,
}


@t.overload
def cast(
    native: Scalar[Any], target_type: DataTypeT, *, safe: bool | None = ...
) -> Scalar[DataTypeT]: ...
@t.overload
def cast(
    native: ChunkedArray[Any], target_type: DataTypeT, *, safe: bool | None = ...
) -> ChunkedArray[Scalar[DataTypeT]]: ...
@t.overload
def cast(
    native: ChunkedOrScalar[Scalar[Any]],
    target_type: DataTypeT,
    *,
    safe: bool | None = ...,
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]: ...
def cast(
    native: ChunkedOrScalar[Scalar[Any]],
    target_type: DataTypeT,
    *,
    safe: bool | None = None,
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]:
    return pc.cast(native, target_type, safe=safe)


def cast_schema(
    native: pa.Schema, target_types: DataType | Mapping[str, DataType] | DataTypeRemap
) -> pa.Schema:
    if isinstance(target_types, pa.DataType):
        return pa.schema((name, target_types) for name in native.names)
    if _is_into_pyarrow_schema(target_types):
        new_schema = native
        for name, dtype in target_types.items():
            index = native.get_field_index(name)
            new_schema.set(index, native.field(index).with_type(dtype))
        return new_schema
    return pa.schema((fld.name, target_types.get(fld.type, fld.type)) for fld in native)


def cast_table(
    native: pa.Table, target: DataType | IntoArrowSchema | DataTypeRemap
) -> pa.Table:
    s = target if isinstance(target, pa.Schema) else cast_schema(native.schema, target)
    return native.cast(s)


def has_large_string(data_types: Iterable[DataType], /) -> bool:
    return any(pa.types.is_large_string(tp) for tp in data_types)


def string_type(data_types: Iterable[DataType] = (), /) -> StringType | LargeStringType:
    """Return a native string type, compatible with `data_types`.

    Until [apache/arrow#45717] is resolved, we need to upcast `string` to `large_string` when joining.

    [apache/arrow#45717]: https://github.com/apache/arrow/issues/45717
    """
    return pa.large_string() if has_large_string(data_types) else pa.string()


def any_(native: Any) -> pa.BooleanScalar:
    return pc.any(native, min_count=0)


def all_(native: Any) -> pa.BooleanScalar:
    return pc.all(native, min_count=0)


def sum_(native: Any) -> NativeScalar:
    return pc.sum(native, min_count=0)


min_ = pc.min
min_horizontal = pc.min_element_wise
max_ = pc.max
max_horizontal = pc.max_element_wise
mean = pc.mean
count = pc.count
median = pc.approximate_median
std = pc.stddev
var = pc.variance
quantile = pc.quantile


def n_unique(native: Any) -> pa.Int64Scalar:
    return count(native, mode="all")


def _reverse(native: ChunkedArrayAny) -> ChunkedArrayAny:
    """Unlike other slicing ops, `[::-1]` creates a full-copy.

    https://github.com/apache/arrow/issues/19103#issuecomment-1377671886
    """
    return native[::-1]


def cumulative(native: ChunkedArrayAny, cum_agg: F.CumAgg, /) -> ChunkedArrayAny:
    func = _CUMULATIVE[type(cum_agg)]
    if not cum_agg.reverse:
        return func(native)
    return _reverse(func(_reverse(native)))


def cum_sum(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return pc.cumulative_sum(native, skip_nulls=True)


def cum_min(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return pc.cumulative_min(native, skip_nulls=True)


def cum_max(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return pc.cumulative_max(native, skip_nulls=True)


def cum_prod(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return pc.cumulative_prod(native, skip_nulls=True)


def cum_count(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return cum_sum(is_not_null(native).cast(pa.uint32()))


_CUMULATIVE: Mapping[type[F.CumAgg], Callable[[ChunkedArrayAny], ChunkedArrayAny]] = {
    F.CumSum: cum_sum,
    F.CumCount: cum_count,
    F.CumMin: cum_min,
    F.CumMax: cum_max,
    F.CumProd: cum_prod,
}


def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT],
    upper: ChunkedOrScalar[ScalarT],
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]:
    fn_lhs, fn_rhs = _IS_BETWEEN[closed]
    return and_(fn_lhs(native, lower), fn_rhs(native, upper))


@t.overload
def is_in(
    values: ChunkedArrayAny, /, other: ChunkedOrArrayAny
) -> ChunkedArray[pa.BooleanScalar]: ...
@t.overload
def is_in(values: ArrayAny, /, other: ChunkedOrArrayAny) -> Array[pa.BooleanScalar]: ...
@t.overload
def is_in(values: ScalarAny, /, other: ChunkedOrArrayAny) -> pa.BooleanScalar: ...
def is_in(values: ArrowAny, /, other: ChunkedOrArrayAny) -> ArrowAny:
    """Check if elements of `values` are present in `other`.

    Roughly equivalent to [`polars.Expr.is_in`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_in.html)

    Returns a mask with `len(values)` elements.
    """
    # NOTE: Stubs don't include a `ChunkedArray` return
    # NOTE: Replaced ambiguous parameter name (`value_set`)
    is_in_: Incomplete = pc.is_in
    return is_in_(values, other)  # type: ignore[no-any-return]


def binary(
    lhs: ChunkedOrScalarAny, op: type[ops.Operator], rhs: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return _DISPATCH_BINARY[op](lhs, rhs)


def concat_str(
    *arrays: ChunkedArrayAny, separator: str = "", ignore_nulls: bool = False
) -> ChunkedArray[StringScalar]:
    dtype = string_type(obj.type for obj in arrays)
    it = (obj.cast(dtype) for obj in arrays)
    concat: Incomplete = pc.binary_join_element_wise
    join = options.join(ignore_nulls=ignore_nulls)
    return concat(*it, lit(separator, dtype), options=join)  # type: ignore[no-any-return]


def int_range(
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    /,
    *,
    dtype: IntegerType = pa.int64(),  # noqa: B008
) -> ChunkedArray[IntegerScalar]:
    import numpy as np  # ignore-banned-import

    if end is None:
        end = start
        start = 0
    return pa.chunked_array([pa.array(np.arange(start, end, step), dtype)])


def lit(value: Any, dtype: DataType | None = None) -> NativeScalar:
    return pa.scalar(value) if dtype is None else pa.scalar(value, dtype)


def array(
    value: NativeScalar | Iterable[Any], dtype: DataType | None = None, /
) -> ArrayAny:
    return (
        pa.array([value], value.type)
        if isinstance(value, pa.Scalar)
        else pa.array(value, dtype)
    )


def chunked_array(
    arr: ArrowAny | list[Iterable[Any]], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    return _chunked_array(array(arr) if isinstance(arr, pa.Scalar) else arr, dtype)


def concat_vertical_chunked(
    arrays: Iterable[ChunkedArrayAny], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    v_concat: Incomplete = pa.chunked_array
    return v_concat(arrays, dtype)  # type: ignore[no-any-return]


def concat_vertical_table(
    tables: Iterable[pa.Table], /, promote_options: PromoteOptions = "none"
) -> pa.Table:
    return pa.concat_tables(tables, promote_options=promote_options)


if BACKEND_VERSION >= (14,):

    def concat_diagonal(tables: Iterable[pa.Table]) -> pa.Table:
        return pa.concat_tables(tables, promote_options="default")
else:

    def concat_diagonal(tables: Iterable[pa.Table]) -> pa.Table:
        return pa.concat_tables(tables, promote=True)


def is_series(obj: t.Any) -> TypeIs[ArrowSeries]:
    from narwhals._plan.arrow.series import ArrowSeries

    return isinstance(obj, ArrowSeries)


def _is_into_pyarrow_schema(obj: Mapping[Any, Any]) -> TypeIs[Mapping[str, DataType]]:
    return (
        (first := next(iter(obj.items())), None)
        and isinstance(first[0], str)
        and isinstance(first[1], pa.DataType)
    )

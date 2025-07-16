"""Native functions, aliased and/or with behavior aligned to `polars`."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import (
    cast_for_truediv,
    chunked_array as _chunked_array,
    floordiv_compat as floordiv,
)
from narwhals._plan import operators as ops

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from narwhals._arrow.typing import (
        ArrayAny,
        ArrayOrScalar,
        ChunkedArrayAny,
        Incomplete,
    )
    from narwhals._plan.arrow.typing import (
        BinaryComp,
        BinaryLogical,
        BinaryNumericTemporal,
        BinOp,
        ChunkedArray,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        DataType,
        DataTypeT,
        NativeScalar,
        Scalar,
        ScalarAny,
        ScalarT,
        StringScalar,
        UnaryFunction,
    )
    from narwhals.typing import ClosedInterval

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


def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT],
    upper: ChunkedOrScalar[ScalarT],
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]:
    fn_lhs, fn_rhs = _IS_BETWEEN[closed]
    return and_(fn_lhs(native, lower), fn_rhs(native, upper))


def binary(
    lhs: ChunkedOrScalarAny, op: type[ops.Operator], rhs: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return _DISPATCH_BINARY[op](lhs, rhs)


def concat_str(
    *arrays: ChunkedArrayAny, separator: str = "", ignore_nulls: bool = False
) -> ChunkedArray[StringScalar]:
    fn: Incomplete = pc.binary_join_element_wise
    it, sep = _cast_to_comparable_string_types(arrays, separator)
    return fn(*it, sep, null_handling="skip" if ignore_nulls else "emit_null")  # type: ignore[no-any-return]


def _cast_to_comparable_string_types(
    arrays: Sequence[ChunkedArrayAny], /, separator: str
) -> tuple[Iterator[ChunkedArray[StringScalar]], StringScalar]:
    # Ensure `chunked_arrays` are either all `string` or all `large_string`.
    dtype = (
        pa.string()
        if not any(pa.types.is_large_string(obj.type) for obj in arrays)
        else pa.large_string()
    )
    return (obj.cast(dtype) for obj in arrays), pa.scalar(separator, dtype)


def lit(value: Any, dtype: DataType | None = None) -> NativeScalar:
    # NOTE: PR that fixed these the overloads was closed
    # https://github.com/zen-xu/pyarrow-stubs/pull/208
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
    arr: ArrayOrScalar | list[Iterable[Any]], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    return _chunked_array(array(arr) if isinstance(arr, pa.Scalar) else arr, dtype)

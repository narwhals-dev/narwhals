from __future__ import annotations

import typing as t
from collections.abc import Callable
from typing import TYPE_CHECKING, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan.arrow.functions._bin_op import and_, gt, gt_eq, lt, lt_eq
from narwhals._plan.arrow.functions._common import MinMax, is_arrow
from narwhals._plan.arrow.functions._construction import array, lit
from narwhals._plan.arrow.functions._dtypes import BOOL

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import TypeAlias

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        Arrow,
        ArrowAny,
        BinaryComp,
        BooleanLengthPreserving,
        BooleanScalar,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrArrayAny,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        ChunkedStruct,
        ScalarAny,
        ScalarT,
        UnaryFunction,
    )
    from narwhals.typing import (
        ClosedInterval,
        NonNestedLiteral,
        NumericLiteral,
        UniqueKeepStrategy,
    )


IntoColumnAgg: TypeAlias = Callable[[str], ir.AggExpr]
"""Helper constructor for single-column aggregations."""


__all__ = [
    "BOOLEAN_LENGTH_PRESERVING",
    "all",
    "any",
    "eq_missing",
    "is_between",
    "is_finite",
    "is_in",
    "is_nan",
    "is_not_nan",
    "is_not_null",
    "is_null",
    "is_only_nulls",
    "not_",
    "unique_keep_boolean_length_preserving",
]


def any(native: Arrow[BooleanScalar], *, ignore_nulls: bool = True) -> pa.BooleanScalar:
    """Return whether any values in `native` are True.

    Arguments:
        native: Boolean-typed arrow data.
        ignore_nulls: If set to `True` (default), null values are ignored.
            If there are no non-null values, the output is `False`.

            If set to `False`, [Kleene logic] is used to deal with nulls;
            if the column contains any null values and no `True` values,
            the output is null.

    [Kleene logic]: https://en.wikipedia.org/wiki/Three-valued_logic
    """
    ca = t.cast("ChunkedArray[pa.BooleanScalar]", native)
    return pc.any(ca, min_count=0, skip_nulls=ignore_nulls)


def all(native: Arrow[BooleanScalar], *, ignore_nulls: bool = True) -> pa.BooleanScalar:
    """Return whether all values in `native` are True.

    Arguments:
        native: Boolean-typed arrow data.
        ignore_nulls: If set to `True` (default), null values are ignored.
            If there are no non-null values, the output is `True`.

            If set to `False`, [Kleene logic] is used to deal with nulls;
            if the column contains any null values and no `False` values,
            the output is null.

    [Kleene logic]: https://en.wikipedia.org/wiki/Three-valued_logic
    """
    ca = t.cast("ChunkedArray[pa.BooleanScalar]", native)
    return pc.all(ca, min_count=0, skip_nulls=ignore_nulls)


is_null = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_null)
is_not_null = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_valid)
is_nan = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_nan)
is_finite = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_finite)
not_ = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.invert)


@overload
def is_not_nan(native: ChunkedArrayAny) -> ChunkedArray[pa.BooleanScalar]: ...
@overload
def is_not_nan(native: ScalarAny) -> pa.BooleanScalar: ...
@overload
def is_not_nan(native: ChunkedOrScalarAny) -> ChunkedOrScalar[pa.BooleanScalar]: ...
@overload
def is_not_nan(native: Arrow[ScalarAny]) -> Arrow[pa.BooleanScalar]: ...
def is_not_nan(native: Arrow[ScalarAny]) -> Arrow[pa.BooleanScalar]:
    return not_(is_nan(native))


def is_only_nulls(native: ChunkedOrArrayAny, *, nan_is_null: bool = False) -> bool:
    """Return True if `native` has 0 non-null values (and optionally include NaN)."""
    return array(native.is_null(nan_is_null=nan_is_null), BOOL).false_count == 0


@overload
def is_between(
    native: ChunkedArray[ScalarT],
    lower: ChunkedOrScalar[ScalarT] | NumericLiteral,
    upper: ChunkedOrScalar[ScalarT] | NumericLiteral,
    closed: ClosedInterval,
) -> ChunkedArray[pa.BooleanScalar]: ...
@overload
def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT] | NumericLiteral,
    upper: ChunkedOrScalar[ScalarT] | NumericLiteral,
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]: ...
def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT] | NumericLiteral,
    upper: ChunkedOrScalar[ScalarT] | NumericLiteral,
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]:
    fn_lhs, fn_rhs = _IS_BETWEEN[closed]
    low, high = (el if is_arrow(el) else lit(el) for el in (lower, upper))
    out: ChunkedOrScalar[pa.BooleanScalar] = and_(
        fn_lhs(native, low), fn_rhs(native, high)
    )
    return out


@overload
def is_in(
    values: ChunkedArrayAny, /, other: ChunkedOrArrayAny
) -> ChunkedArray[pa.BooleanScalar]: ...
@overload
def is_in(values: ArrayAny, /, other: ChunkedOrArrayAny) -> Array[pa.BooleanScalar]: ...
@overload
def is_in(values: ScalarAny, /, other: ChunkedOrArrayAny) -> pa.BooleanScalar: ...
@overload
def is_in(
    values: ChunkedOrScalarAny, /, other: ChunkedOrArrayAny
) -> ChunkedOrScalarAny: ...
def is_in(values: ArrowAny, /, other: ChunkedOrArrayAny) -> ArrowAny:
    """Check if elements of `values` are present in `other`.

    Roughly equivalent to [`polars.Expr.is_in`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_in.html)

    Returns a mask with `len(values)` elements.
    """
    # NOTE: Stubs don't include a `ChunkedArray` return
    # NOTE: Replaced ambiguous parameter name (`value_set`)
    is_in_: Incomplete = pc.is_in
    return is_in_(values, other)  # type: ignore[no-any-return]


@overload
def eq_missing(
    native: ChunkedArrayAny, other: NonNestedLiteral | ArrowAny
) -> ChunkedArray[pa.BooleanScalar]: ...
@overload
def eq_missing(
    native: ArrayAny, other: NonNestedLiteral | ArrowAny
) -> Array[pa.BooleanScalar]: ...
@overload
def eq_missing(
    native: ScalarAny, other: NonNestedLiteral | ArrowAny
) -> pa.BooleanScalar: ...
@overload
def eq_missing(
    native: ChunkedOrScalarAny, other: NonNestedLiteral | ArrowAny
) -> ChunkedOrScalarAny: ...
def eq_missing(native: ArrowAny, other: NonNestedLiteral | ArrowAny) -> ArrowAny:
    """Equivalent to `native == other` where `None == None`.

    This differs from default `eq` where null values are propagated.

    Note:
        Unique to `pyarrow`, this wrapper will ensure `None` uses `native.type`.
    """
    if isinstance(other, (pa.Array, pa.ChunkedArray)):
        return is_in(native, other)
    item = array(other if isinstance(other, pa.Scalar) else lit(other, native.type))
    return is_in(native, item)


def unique_keep_boolean_length_preserving(
    keep: UniqueKeepStrategy,
) -> tuple[IntoColumnAgg, BooleanLengthPreserving]:
    return BOOLEAN_LENGTH_PRESERVING[_UNIQUE_KEEP_BOOLEAN_LENGTH_PRESERVING[keep]]


def _ir_min_max(name: str, /) -> MinMax:
    return MinMax(expr=ir.col(name))


def _boolean_is_unique(
    indices: ChunkedArrayAny, aggregated: ChunkedStruct, /
) -> ChunkedArrayAny:
    min, max = aggregated.flatten()
    return and_(is_in(indices, min), is_in(indices, max))


def _boolean_is_duplicated(
    indices: ChunkedArrayAny, aggregated: ChunkedStruct, /
) -> ChunkedArrayAny:
    return not_(_boolean_is_unique(indices, aggregated))


# TODO @dangotbanned: Replace with a function for export?
BOOLEAN_LENGTH_PRESERVING: Mapping[
    type[ir.boolean.BooleanFunction], tuple[IntoColumnAgg, BooleanLengthPreserving]
] = {
    ir.boolean.IsFirstDistinct: (ir.min, is_in),
    ir.boolean.IsLastDistinct: (ir.max, is_in),
    ir.boolean.IsUnique: (_ir_min_max, _boolean_is_unique),
    ir.boolean.IsDuplicated: (_ir_min_max, _boolean_is_duplicated),
}

_UNIQUE_KEEP_BOOLEAN_LENGTH_PRESERVING: Mapping[
    UniqueKeepStrategy, type[ir.boolean.BooleanFunction]
] = {
    "any": ir.boolean.IsFirstDistinct,
    "first": ir.boolean.IsFirstDistinct,
    "last": ir.boolean.IsLastDistinct,
    "none": ir.boolean.IsUnique,
}

_IS_BETWEEN: Mapping[ClosedInterval, tuple[BinaryComp, BinaryComp]] = {
    "left": (gt_eq, lt),
    "right": (gt, lt_eq),
    "none": (gt, lt),
    "both": (gt_eq, lt_eq),
}

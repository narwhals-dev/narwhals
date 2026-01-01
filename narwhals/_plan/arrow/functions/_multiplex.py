"""Selection, combination and replacement functions.

Note:
    Not 100% sure on the name yet

- https://arrow.apache.org/docs/python/api/compute.html#selecting-multiplexing
- https://arrow.apache.org/docs/python/api/compute.html#structural-transforms
- https://arrow.apache.org/docs/python/api/compute.html#selections
"""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan._guards import is_non_nested_literal
from narwhals._plan.arrow.functions._bin_op import and_, gt, not_eq, or_, sub
from narwhals._plan.arrow.functions._boolean import any_, is_not_nan, is_not_null, is_null
from narwhals._plan.arrow.functions._common import reverse
from narwhals._plan.arrow.functions._construction import array, chunked_array, lit
from narwhals._plan.arrow.functions._cumulative import cum_max
from narwhals._plan.arrow.functions._ranges import int_range

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        ArrowAny,
        ArrowT,
        BooleanScalar,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrArrayAny,
        ChunkedOrArrayT,
        ChunkedOrScalarAny,
        ChunkedOrScalarT,
        Predicate,
        SameArrowT,
        ScalarAny,
        UnaryFunction,
        VectorFunction,
    )
    from narwhals._plan.typing import Seq
    from narwhals.typing import FillNullStrategy, NonNestedLiteral


__all__ = [
    "drop_nulls",
    "fill_nan",
    "fill_null",
    "fill_null_with_strategy",
    "preserve_nulls",
    "replace_strict",
    "replace_strict_default",
    "replace_with_mask",
    "when_then",
]

drop_nulls = t.cast("VectorFunction[...]", pc.drop_null)
"""Drop all null values.

The original order of the remaining elements is preserved.
"""


@overload
def when_then(
    predicate: ChunkedArray[BooleanScalar], then: ScalarAny
) -> ChunkedArrayAny: ...
@overload
def when_then(predicate: Array[BooleanScalar], then: ScalarAny) -> ArrayAny: ...
@overload
def when_then(
    predicate: Predicate, then: SameArrowT, otherwise: SameArrowT | None
) -> SameArrowT: ...
@overload
def when_then(predicate: Predicate, then: ScalarAny, otherwise: ArrowT) -> ArrowT: ...
@overload
def when_then(
    predicate: Predicate, then: ArrowT, otherwise: ScalarAny | NonNestedLiteral = ...
) -> ArrowT: ...
@overload
def when_then(
    predicate: Predicate, then: ArrowAny, otherwise: ArrowAny | NonNestedLiteral = None
) -> Incomplete: ...
def when_then(
    predicate: Predicate, then: ArrowAny, otherwise: ArrowAny | NonNestedLiteral = None
) -> Incomplete:
    """Return elements from `then` or `otherwise` depending on `predicate`.

    Thin wrapper around [`pc.if_else`], with two tweaks + *some* typing:
    - Supports a 2-argument form, like `pl.when(...).then(...)`
    - Accepts python literals, but only in the `otherwise` position

    [`pc.if_else`]: https://arrow.apache.org/docs/python/generated/pyarrow.compute.if_else.html
    """
    if is_non_nested_literal(otherwise):
        otherwise = lit(otherwise, then.type)
    return pc.if_else(predicate, then, otherwise)


@overload
def replace_with_mask(
    native: ChunkedOrArrayT, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayT: ...
@overload
def replace_with_mask(
    native: ChunkedOrArrayAny, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayAny: ...
def replace_with_mask(
    native: ChunkedOrArrayAny, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayAny:
    """Replace elements of `native`, at positions defined by `mask`.

    The length of `replacements` must equal the number of `True` values in `mask`.
    """
    if isinstance(native, pa.ChunkedArray):
        args = [array(p) for p in (native, mask, replacements)]
        return chunked_array(pc.call_function("replace_with_mask", args))
    args = [native, array(mask), array(replacements)]
    result: ChunkedOrArrayAny = pc.call_function("replace_with_mask", args)
    return result


def replace_strict(
    native: ChunkedOrScalarAny,
    old: Seq[Any],
    new: Seq[Any],
    dtype: pa.DataType | None = None,
) -> ChunkedOrScalarAny:
    """Replace all values (`old`) by different values (`new`).

    Raises if any values in `native` were not replaced.
    """
    if isinstance(native, pa.Scalar):
        idxs: ArrayAny = array(pc.index_in(native, pa.array(old)))
        result: ChunkedOrScalarAny = pa.array(new).take(idxs)[0]
    else:
        idxs = pc.index_in(native, pa.array(old))
        result = chunked_array(pa.array(new).take(idxs))
    if err := _ensure_all_replaced(native, and_(is_not_null(native), is_null(idxs))):
        raise err
    return result.cast(dtype) if dtype else result


def replace_strict_default(
    native: ChunkedOrScalarAny,
    old: Seq[Any],
    new: Seq[Any],
    default: ChunkedOrScalarAny,
    dtype: pa.DataType | None = None,
) -> ChunkedOrScalarAny:
    """Replace all values (`old`) by different values (`new`).

    Sets any values that were not replaced in `native` to `default`.
    """
    idxs = pc.index_in(native, pa.array(old))
    result = pa.array(new).take(array(idxs))
    result = when_then(is_null(idxs), default, result.cast(dtype) if dtype else result)
    return chunked_array(result) if isinstance(native, pa.ChunkedArray) else result[0]


def preserve_nulls(
    before: ChunkedOrArrayAny, after: ChunkedOrArrayT, /
) -> ChunkedOrArrayT:
    """Propagate nulls positionally from `before` to `after`."""
    return when_then(is_not_null(before), after) if before.null_count else after


@t.overload
def fill_null(
    native: ChunkedOrScalarT, value: NonNestedLiteral | ArrowAny
) -> ChunkedOrScalarT: ...
@t.overload
def fill_null(
    native: ChunkedOrArrayT, value: ScalarAny | NonNestedLiteral | ChunkedOrArrayT
) -> ChunkedOrArrayT: ...
@t.overload
def fill_null(
    native: ChunkedOrScalarAny, value: ChunkedOrScalarAny | NonNestedLiteral
) -> ChunkedOrScalarAny: ...
def fill_null(native: ArrowAny, value: ArrowAny | NonNestedLiteral) -> ArrowAny:
    """Fill null values with `value`."""
    fill_value: Incomplete = value
    result: ArrowAny = pc.fill_null(native, fill_value)
    return result


@t.overload
def fill_nan(
    native: ChunkedOrScalarT, value: NonNestedLiteral | ArrowAny
) -> ChunkedOrScalarT: ...
@t.overload
def fill_nan(native: SameArrowT, value: NonNestedLiteral | ArrowAny) -> SameArrowT: ...
def fill_nan(native: ArrowAny, value: NonNestedLiteral | ArrowAny) -> Incomplete:
    """Fill floating point NaN values with `value`."""
    return when_then(is_not_nan(native), native, value)


def fill_null_with_strategy(
    native: ChunkedArrayAny, strategy: FillNullStrategy, limit: int | None = None
) -> ChunkedArrayAny:
    """Fill null values with `strategy`, optionally to at most `limit` consecutive null values."""
    null_count = native.null_count
    if null_count == 0 or (null_count == len(native)):
        return native
    if limit is None:
        return _FILL_NULL_STRATEGY[strategy](native)
    if strategy == "forward":
        return _fill_null_forward_limit(native, limit)
    return reverse(_fill_null_forward_limit(reverse(native), limit))


def _ensure_all_replaced(
    native: ChunkedOrScalarAny, unmatched: ArrowAny
) -> ValueError | None:
    if not any_(unmatched).as_py():
        return None
    msg = (
        "replace_strict did not replace all non-null values.\n\n"
        f"The following did not get replaced: {chunked_array(native).filter(array(unmatched)).unique().to_pylist()}"
    )
    return ValueError(msg)


def _fill_null_forward_limit(native: ChunkedArrayAny, limit: int) -> ChunkedArrayAny:
    SENTINEL = lit(-1)  # noqa: N806
    is_not_null = native.is_valid()
    index = int_range(len(native), chunked=False)
    index_not_null = cum_max(when_then(is_not_null, index, SENTINEL))
    # NOTE: The correction here is for nulls at either end of the array
    # They should be preserved when the `strategy` would need an out-of-bounds index
    not_oob = not_eq(index_not_null, SENTINEL)
    index_not_null = when_then(not_oob, index_not_null)
    beyond_limit = gt(sub(index, index_not_null), lit(limit))
    return when_then(or_(is_not_null, beyond_limit), native, native.take(index_not_null))


_FILL_NULL_STRATEGY: Mapping[FillNullStrategy, UnaryFunction] = {
    "forward": pc.fill_null_forward,
    "backward": pc.fill_null_backward,
}

from __future__ import annotations

from typing import TYPE_CHECKING, get_args, overload

from narwhals._plan import _guards, selectors as cs
from narwhals._plan._namespace import namespace
from narwhals._plan.compliant.concat import can_concat_dataframe
from narwhals._plan.exceptions import unsupported_backend_operation_error
from narwhals._plan.functions.aggregation import max, mean, median, min, sum
from narwhals._plan.functions.col import col
from narwhals._plan.functions.horizontal import (
    all_horizontal,
    any_horizontal,
    coalesce,
    concat_str,
    format,
    max_horizontal,
    mean_horizontal,
    min_horizontal,
    sum_horizontal,
)
from narwhals._plan.functions.len import len
from narwhals._plan.functions.literal import lit
from narwhals._plan.functions.ranges import date_range, int_range, linear_space
from narwhals.typing import ConcatMethod

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._plan.expr import Expr
    from narwhals._plan.typing import DataFrameT, LazyFrameT, OneOrIterable


__all__ = (
    "all",
    "all_horizontal",
    "any_horizontal",
    "coalesce",
    "col",
    "concat",
    "concat_str",
    "date_range",
    "exclude",
    "format",
    "int_range",
    "len",
    "linear_space",
    "lit",
    "max",
    "max_horizontal",
    "mean",
    "mean_horizontal",
    "median",
    "min",
    "min_horizontal",
    "nth",
    "sum",
    "sum_horizontal",
)


def nth(*indices: OneOrIterable[int]) -> Expr:
    return cs.by_index(*indices).as_expr()


def all() -> Expr:
    return cs.all().as_expr()


def exclude(*names: OneOrIterable[str]) -> Expr:
    return cs.all().exclude(*names).as_expr()


@overload
def concat(
    items: Iterable[DataFrameT], *, how: ConcatMethod = "vertical"
) -> DataFrameT: ...
@overload
def concat(
    items: Iterable[LazyFrameT], *, how: ConcatMethod = "vertical"
) -> LazyFrameT: ...
def concat(
    items: Iterable[DataFrameT] | Iterable[LazyFrameT], *, how: ConcatMethod = "vertical"
) -> DataFrameT | LazyFrameT:
    frames: Sequence[DataFrameT | LazyFrameT] = tuple(items)
    if not frames:
        msg = "Cannot concatenate an empty iterable."
        raise ValueError(msg)
    if how not in {"horizontal", "vertical", "diagonal"}:
        msg = f"Only the following concatenation methods are supported: {get_args(ConcatMethod)}; found '{how}'."
        raise NotImplementedError(msg)
    return _concat(frames, how)


def _concat(
    frames: Sequence[DataFrameT | LazyFrameT], /, how: ConcatMethod
) -> DataFrameT | LazyFrameT:
    if _guards.is_lazyframe(frames[0]):
        if _guards.is_sequence_lazyframe(frames):
            return _concat_lazy(frames, how)

    elif _guards.is_sequence_dataframe(frames):
        return _concat_eager(frames, how)

    msg = f"The items to concatenate should either all be eager, or all lazy, got: {[type(item) for item in frames]}"
    raise TypeError(msg)


def _concat_lazy(frames: Sequence[LazyFrameT], how: ConcatMethod) -> LazyFrameT:
    from narwhals._plan.plans import logical

    return frames[0]._with_lp(logical.concat(tuple(lf._plan for lf in frames), how=how))


def _concat_eager(frames: Sequence[DataFrameT], how: ConcatMethod) -> DataFrameT:
    ns = namespace(frames[0])
    if not can_concat_dataframe(ns):
        raise unsupported_backend_operation_error(ns.implementation, "concat")
    return frames[0]._with_compliant(ns.concat_df((df._compliant for df in frames), how))

from __future__ import annotations

import builtins
import typing as t
from typing import TYPE_CHECKING, get_args

from narwhals._plan import selectors as cs
from narwhals._plan.compliant.typing import namespace
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
    from collections.abc import Iterable

    from narwhals._plan.expr import Expr
    from narwhals._plan.typing import DataFrameT, OneOrIterable

    T = t.TypeVar("T")


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


# TODO @dangotbanned: Update this when `LazyFrame` exists
def concat(items: Iterable[DataFrameT], *, how: ConcatMethod = "vertical") -> DataFrameT:
    elems = list(items)
    if not elems:
        msg = "Cannot concatenate an empty iterable."
        raise ValueError(msg)
    if how not in {"horizontal", "vertical", "diagonal"}:
        msg = f"Only the following concatenation methods are supported: {get_args(ConcatMethod)}; found '{how}'."
        raise NotImplementedError(msg)
    elems = _ensure_same_frame(elems)
    compliant = namespace(elems[0]).concat((df._compliant for df in elems), how=how)
    return elems[0]._with_compliant(compliant)


def _ensure_same_frame(items: list[T], /) -> list[T]:
    item_0_tp = type(items[0])
    if builtins.all(isinstance(item, item_0_tp) for item in items):
        return items
    msg = f"The items to concatenate should either all be eager, or all lazy, got: {[type(item) for item in items]}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover

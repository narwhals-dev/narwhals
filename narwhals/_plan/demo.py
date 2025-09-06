from __future__ import annotations

import builtins
import typing as t

from narwhals._plan import boolean, expr, expr_parsing as parse, functions as F
from narwhals._plan.common import (
    into_dtype,
    is_non_nested_literal,
    is_series,
    py_to_narwhals_dtype,
)
from narwhals._plan.expr import All, Len
from narwhals._plan.literal import ScalarLiteral, SeriesLiteral
from narwhals._plan.ranges import IntRange
from narwhals._plan.strings import ConcatHorizontal
from narwhals._plan.when_then import When
from narwhals._utils import Version, flatten

if t.TYPE_CHECKING:
    from narwhals._plan.dummy import Expr, Series
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, NativeSeriesT
    from narwhals.dtypes import IntegerType
    from narwhals.typing import IntoDType, NonNestedLiteral


def col(*names: str | t.Iterable[str]) -> Expr:
    flat = tuple(flatten(names))
    node = expr.col(flat[0]) if builtins.len(flat) == 1 else expr.cols(*flat)
    return node.to_narwhals()


def nth(*indices: int | t.Sequence[int]) -> Expr:
    flat = tuple(flatten(indices))
    node = expr.nth(flat[0]) if builtins.len(flat) == 1 else expr.index_columns(*flat)
    return node.to_narwhals()


def lit(
    value: NonNestedLiteral | Series[NativeSeriesT], dtype: IntoDType | None = None
) -> Expr:
    if is_series(value):
        return SeriesLiteral(value=value).to_literal().to_narwhals()
    if not is_non_nested_literal(value):
        msg = f"{type(value).__name__!r} is not supported in `nw.lit`, got: {value!r}."
        raise TypeError(msg)
    if dtype is None:
        dtype = py_to_narwhals_dtype(value, Version.MAIN)
    else:
        dtype = into_dtype(dtype)
    return ScalarLiteral(value=value, dtype=dtype).to_literal().to_narwhals()


def len() -> Expr:
    return Len().to_narwhals()


def all() -> Expr:
    return All().to_narwhals()


def exclude(*names: str | t.Iterable[str]) -> Expr:
    return all().exclude(*names)


def max(*columns: str) -> Expr:
    return col(columns).max()


def mean(*columns: str) -> Expr:
    return col(columns).mean()


def min(*columns: str) -> Expr:
    return col(columns).min()


def median(*columns: str) -> Expr:
    return col(columns).median()


def sum(*columns: str) -> Expr:
    return col(columns).sum()


def all_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return boolean.AllHorizontal().to_function_expr(*it).to_narwhals()


def any_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return boolean.AnyHorizontal().to_function_expr(*it).to_narwhals()


def sum_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.SumHorizontal().to_function_expr(*it).to_narwhals()


def min_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MinHorizontal().to_function_expr(*it).to_narwhals()


def max_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MaxHorizontal().to_function_expr(*it).to_narwhals()


def mean_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MeanHorizontal().to_function_expr(*it).to_narwhals()


def concat_str(
    exprs: IntoExpr | t.Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    it = parse.parse_into_seq_of_expr_ir(exprs, *more_exprs)
    return (
        ConcatHorizontal(separator=separator, ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )


def when(
    *predicates: IntoExprColumn | t.Iterable[IntoExprColumn], **constraints: t.Any
) -> When:
    """Start a `when-then-otherwise` expression.

    Examples:
        >>> from narwhals._plan import demo as nwd

        >>> nwd.when(nwd.col("y") == "b").then(1)
        nw._plan.Expr(main):
        .when([(col('y')) == (lit(str: b))]).then(lit(int: 1)).otherwise(lit(null))
    """
    condition = parse.parse_predicates_constraints_into_expr_ir(
        *predicates, **constraints
    )
    return When._from_ir(condition)


def int_range(
    start: int | IntoExprColumn = 0,
    end: int | IntoExprColumn | None = None,
    step: int = 1,
    *,
    dtype: IntegerType | type[IntegerType] = Version.MAIN.dtypes.Int64,
    eager: bool = False,
) -> Expr:
    if end is None:
        end = start
        start = 0
    if eager:
        msg = f"{eager=}"
        raise NotImplementedError(msg)
    return (
        IntRange(step=step, dtype=into_dtype(dtype))
        .to_function_expr(*parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )

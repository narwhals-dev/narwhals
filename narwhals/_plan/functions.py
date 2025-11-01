from __future__ import annotations

import builtins
import typing as t

from narwhals._plan import _guards, _parse, common, expressions as ir, selectors as cs
from narwhals._plan.expressions import functions as F
from narwhals._plan.expressions.literal import ScalarLiteral, SeriesLiteral
from narwhals._plan.expressions.ranges import IntRange
from narwhals._plan.expressions.strings import ConcatStr
from narwhals._plan.when_then import When
from narwhals._utils import Version, flatten

if t.TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, NativeSeriesT
    from narwhals.dtypes import IntegerType
    from narwhals.typing import IntoDType, NonNestedLiteral


def col(*names: str | t.Iterable[str]) -> Expr:
    flat = tuple(flatten(names))
    return (
        ir.col(flat[0]).to_narwhals()
        if builtins.len(flat) == 1
        else cs.by_name(*flat).as_expr()
    )


def nth(*indices: int | t.Sequence[int]) -> Expr:
    return cs.by_index(*indices).as_expr()


def lit(
    value: NonNestedLiteral | Series[NativeSeriesT], dtype: IntoDType | None = None
) -> Expr:
    if _guards.is_series(value):
        return SeriesLiteral(value=value).to_literal().to_narwhals()
    if not _guards.is_non_nested_literal(value):
        msg = f"{type(value).__name__!r} is not supported in `nw.lit`, got: {value!r}."
        raise TypeError(msg)
    if dtype is None:
        dtype = common.py_to_narwhals_dtype(value, Version.MAIN)
    else:
        dtype = common.into_dtype(dtype)
    return ScalarLiteral(value=value, dtype=dtype).to_literal().to_narwhals()


def len() -> Expr:
    return ir.Len().to_narwhals()


def all() -> Expr:
    return cs.all().as_expr()


def exclude(*names: str | t.Iterable[str]) -> Expr:
    return cs.all().exclude(*names).as_expr()


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
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return ir.boolean.AllHorizontal().to_function_expr(*it).to_narwhals()


def any_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return ir.boolean.AnyHorizontal().to_function_expr(*it).to_narwhals()


def sum_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.SumHorizontal().to_function_expr(*it).to_narwhals()


def min_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MinHorizontal().to_function_expr(*it).to_narwhals()


def max_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MaxHorizontal().to_function_expr(*it).to_narwhals()


def mean_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MeanHorizontal().to_function_expr(*it).to_narwhals()


def concat_str(
    exprs: IntoExpr | t.Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(exprs, *more_exprs)
    return (
        ConcatStr(separator=separator, ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )


def when(
    *predicates: IntoExprColumn | t.Iterable[IntoExprColumn], **constraints: t.Any
) -> When:
    """Start a `when-then-otherwise` expression.

    Examples:
        >>> from narwhals import _plan as nw

        >>> nw.when(nw.col("y") == "b").then(1)
        nw._plan.Expr(main):
        .when([(col('y')) == (lit(str: b))]).then(lit(int: 1)).otherwise(lit(null))
    """
    condition = _parse.parse_predicates_constraints_into_expr_ir(
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
        IntRange(step=step, dtype=common.into_dtype(dtype))
        .to_function_expr(*_parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )

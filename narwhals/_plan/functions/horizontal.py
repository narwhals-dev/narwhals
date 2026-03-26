from __future__ import annotations

from typing import TYPE_CHECKING, Final

from narwhals._plan import _parse, expressions as ir
from narwhals._plan.expressions import functions as F
from narwhals._plan.expressions.strings import ConcatStr
from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals._plan.typing import IntoExpr, OneOrIterable

__all__ = [
    "all_horizontal",
    "any_horizontal",
    "coalesce",
    "concat_str",
    "format",
    "max_horizontal",
    "mean_horizontal",
    "min_horizontal",
    "sum_horizontal",
]

_string: Final = Version.MAIN.dtypes.String()


def all_horizontal(*exprs: OneOrIterable[IntoExpr], ignore_nulls: bool = False) -> Expr:
    it = _parse.into_seq_of_expr_ir(*exprs)
    return ir.boolean.all_horizontal(*it, ignore_nulls=ignore_nulls).to_narwhals()


def any_horizontal(*exprs: OneOrIterable[IntoExpr], ignore_nulls: bool = False) -> Expr:
    it = _parse.into_seq_of_expr_ir(*exprs)
    return (
        ir.boolean.AnyHorizontal(ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )


def coalesce(exprs: OneOrIterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    it = _parse.into_seq_of_expr_ir(exprs, *more_exprs)
    return F.Coalesce().to_function_expr(*it).to_narwhals()


def concat_str(
    exprs: OneOrIterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    it = _parse.into_seq_of_expr_ir(exprs, *more_exprs)
    return (
        ConcatStr(separator=separator, ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )


def format(f_string: str, *args: IntoExpr) -> Expr:
    """Format expressions as a string.

    Arguments:
        f_string: A string that with placeholders.
        args: Expression(s) that fill the placeholders.
    """
    if (n_placeholders := f_string.count("{}")) != len(args):
        msg = f"number of placeholders should equal the number of arguments. Expected {n_placeholders} arguments, got {len(args)}."
        raise ValueError(msg)
    exprs: list[ir.ExprIR] = []
    it = iter(args)
    for i, s in enumerate(f_string.split("{}")):
        if i > 0:
            exprs.append(_parse.into_expr_ir(next(it)))
        if s:
            exprs.append(ir.lit(s, _string))
    f = ConcatStr(separator="", ignore_nulls=False)
    return f.to_function_expr(*exprs).to_narwhals()


def max_horizontal(*exprs: OneOrIterable[IntoExpr]) -> Expr:
    it = _parse.into_seq_of_expr_ir(*exprs)
    return F.MaxHorizontal().to_function_expr(*it).to_narwhals()


def mean_horizontal(*exprs: OneOrIterable[IntoExpr]) -> Expr:
    it = _parse.into_seq_of_expr_ir(*exprs)
    return F.MeanHorizontal().to_function_expr(*it).to_narwhals()


def min_horizontal(*exprs: OneOrIterable[IntoExpr]) -> Expr:
    it = _parse.into_seq_of_expr_ir(*exprs)
    return F.MinHorizontal().to_function_expr(*it).to_narwhals()


def sum_horizontal(*exprs: OneOrIterable[IntoExpr]) -> Expr:
    it = _parse.into_seq_of_expr_ir(*exprs)
    return F.SumHorizontal().to_function_expr(*it).to_narwhals()

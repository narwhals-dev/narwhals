from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._guards import is_function_expr, is_window_expr
from narwhals._plan.exceptions import (
    over_elementwise_error as elementwise_error,
    over_nested_error as nested_error,
    over_row_separable_error as row_separable_error,
)
from narwhals._plan.expressions.expr import OrderedWindowExpr, WindowExpr
from narwhals._plan.options import SortOptions

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR
    from narwhals._plan.typing import Seq


def _validate_over(
    expr: ExprIR,
    partition_by: Seq[ExprIR],
    order_by: Seq[ExprIR] = (),
    sort_options: SortOptions | None = None,
    /,
) -> ValueError | None:
    if is_window_expr(expr):
        return nested_error(expr, partition_by, order_by, sort_options)
    if is_function_expr(expr):
        if expr.options.is_elementwise():
            return elementwise_error(expr, partition_by, order_by, sort_options)
        if expr.options.is_row_separable():
            return row_separable_error(expr, partition_by, order_by, sort_options)
    return None


def over(expr: ExprIR, partition_by: Seq[ExprIR], /) -> WindowExpr:
    if err := _validate_over(expr, partition_by):
        raise err
    return WindowExpr(expr=expr, partition_by=partition_by)


def over_ordered(
    expr: ExprIR,
    partition_by: Seq[ExprIR],
    order_by: Seq[ExprIR],
    /,
    *,
    descending: bool = False,
    nulls_last: bool = False,
) -> OrderedWindowExpr:
    sort_options = SortOptions(descending=descending, nulls_last=nulls_last)
    if err := _validate_over(expr, partition_by, order_by, sort_options):
        raise err
    return OrderedWindowExpr(
        expr=expr, partition_by=partition_by, order_by=order_by, sort_options=sort_options
    )

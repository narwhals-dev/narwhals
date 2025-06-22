from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Immutable, is_function_expr, is_window_expr
from narwhals._plan.exceptions import (
    over_elementwise_error,
    over_nested_error,
    over_row_separable_error,
)

if TYPE_CHECKING:
    from narwhals._plan.common import ExprIR
    from narwhals._plan.expr import OrderedWindowExpr, WindowExpr
    from narwhals._plan.options import SortOptions
    from narwhals._plan.typing import Seq
    from narwhals.exceptions import InvalidOperationError


class Window(Immutable):
    """Renamed from `WindowType`.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/options/mod.rs#L139
    """


class Over(Window):
    @staticmethod
    def _validate_over(
        expr: ExprIR,
        partition_by: Seq[ExprIR],
        order_by: Seq[ExprIR] = (),
        sort_options: SortOptions | None = None,
        /,
    ) -> InvalidOperationError | None:
        if is_window_expr(expr):
            return over_nested_error(expr, partition_by, order_by, sort_options)
        if is_function_expr(expr):
            if expr.options.is_elementwise():
                return over_elementwise_error(expr, partition_by, order_by, sort_options)
            if expr.options.is_row_separable():
                return over_row_separable_error(
                    expr, partition_by, order_by, sort_options
                )
        return None

    def to_window_expr(self, expr: ExprIR, partition_by: Seq[ExprIR], /) -> WindowExpr:
        from narwhals._plan.expr import WindowExpr

        if err := self._validate_over(expr, partition_by):
            raise err
        return WindowExpr(expr=expr, partition_by=partition_by, options=self)

    def to_ordered_window_expr(
        self,
        expr: ExprIR,
        partition_by: Seq[ExprIR],
        order_by: Seq[ExprIR],
        sort_options: SortOptions,
        /,
    ) -> OrderedWindowExpr:
        from narwhals._plan.expr import OrderedWindowExpr

        if err := self._validate_over(expr, partition_by, order_by, sort_options):
            raise err
        return OrderedWindowExpr(
            expr=expr,
            partition_by=partition_by,
            order_by=order_by,
            sort_options=sort_options,
            options=self,
        )

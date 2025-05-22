from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Immutable

if TYPE_CHECKING:
    from narwhals._plan.common import ExprIR, Seq
    from narwhals._plan.expr import WindowExpr
    from narwhals._plan.options import SortOptions


class Window(Immutable):
    """Renamed from `WindowType`.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/options/mod.rs#L139
    """


# TODO @dangotbanned: What are all the variants we have code paths for?
# - Over has *at least* (partition_by,), (order_by,), (partition_by, order_by), + options
# - `_plan.expr.WindowExpr` has:
#    - expr (last node)
#    - partition_by, optional order_by, `options` which is one of these classes?
class Over(Window):
    def to_window_expr(
        self,
        expr: ExprIR,
        partition_by: Seq[ExprIR],
        order_by: tuple[Seq[ExprIR], SortOptions] | None,
        /,
    ) -> WindowExpr:
        from narwhals._plan.expr import WindowExpr

        return WindowExpr(
            expr=expr, partition_by=partition_by, order_by=order_by, options=self
        )

"""TODO: Figure out what `Over` should be holding or skip it and go straight to `WindowExpr`."""

from __future__ import annotations

from narwhals._plan.common import ExprIR


class Window(ExprIR):
    """Renamed from `WindowType`.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/options/mod.rs#L139
    """


# TODO @dangotbanned: What are all the variants we have code paths for?
# - Over has *at least* (partition_by,), (order_by,), (partition_by, order_by), + options
# - `_plan.expr.WindowExpr` has:
#    - expr (last node)
#    - partition_by, optional order_by, `options` which is one of these classes?
class Over(Window): ...

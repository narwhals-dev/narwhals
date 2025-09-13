"""Post-`_expansion` rewrites, in a similar style."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._expansion import into_named_irs, prepare_projection
from narwhals._plan._guards import (
    is_aggregation,
    is_binary_expr,
    is_function_expr,
    is_window_expr,
)
from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals._plan.common import NamedIR, map_ir, replace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._plan.common import ExprIR
    from narwhals._plan.schema import IntoFrozenSchema
    from narwhals._plan.typing import IntoExpr, MapIR, Seq


def rewrite_all(
    *exprs: IntoExpr, schema: IntoFrozenSchema, rewrites: Sequence[MapIR]
) -> Seq[NamedIR]:
    """Very naive approach, but should work for a demo.

    - Applying multiple functions should be happening at a lower level
      - Currently we do a full traversal of each tree per-rewrite function
    - There's no caching *after* `prepare_projection` yet
    """
    out_irs, _, names = prepare_projection(parse_into_seq_of_expr_ir(*exprs), schema)
    named_irs = into_named_irs(out_irs, names)
    return tuple(map_ir(ir, *rewrites) for ir in named_irs)


def rewrite_elementwise_over(window: ExprIR, /) -> ExprIR:
    """Requested in [discord-0].

    Before:

        nw.col("a").sum().abs().over("b")

    After:

        nw.col("a").sum().over("b").abs()

    [discord-0]: https://discord.com/channels/1235257048170762310/1383078215303696544/1384807793512677398
    """
    if (
        is_window_expr(window)
        and is_function_expr(window.expr)
        and window.expr.options.is_elementwise()
    ):
        func = window.expr
        parent, *args = func.input
        return replace(func, input=(replace(window, expr=parent), *args))
    return window


# TODO @dangotbanned: Tests (single ✔️, multiple ✔️, complex ❌)
def rewrite_binary_agg_over(window: ExprIR, /) -> ExprIR:
    """Requested in [discord-1], clarified in [discord-2].

    Before:

        (nw.col("a") - nw.col("a").mean()).over("b")

    After:

        nw.col("a") - nw.col("a").mean().over("b")

    [discord-1]: https://discord.com/channels/1235257048170762310/1383078215303696544/1384850753008435372
    [discord-2]: https://discord.com/channels/1235257048170762310/1383078215303696544/1384869107203047588
    """
    if (
        is_window_expr(window)
        and is_binary_expr(window.expr)
        and (is_aggregation(window.expr.right))
    ):
        binary_expr = window.expr
        return replace(binary_expr, right=replace(window, expr=binary_expr.right))
    return window

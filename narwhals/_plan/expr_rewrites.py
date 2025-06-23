"""Post-`expr_expansion` rewrites, in a similar style."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan import expr_parsing as parse
from narwhals._plan.common import NamedIR, is_function_expr, is_window_expr, map_ir
from narwhals._plan.expr_expansion import into_named_irs, prepare_projection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals._plan.common import ExprIR
    from narwhals._plan.typing import IntoExpr, MapIR, Seq
    from narwhals.dtypes import DType


def rewrite_all(
    *exprs: IntoExpr, schema: Mapping[str, DType], rewrites: Sequence[MapIR]
) -> Seq[NamedIR]:
    """Very naive approach, but should work for a demo.

    - Applying multiple functions should be happening at a lower level
      - Currently we do a full traversal of each tree per-rewrite function
    - There's no caching *after* `prepare_projection` yet
    """
    out_irs, _, names = prepare_projection(
        parse.parse_into_seq_of_expr_ir(*exprs), schema
    )
    named_irs = into_named_irs(out_irs, names)
    return tuple(map_ir(ir, *rewrites) for ir in named_irs)


# TODO @dangotbanned: Tests
# TODO @dangotbanned: Review if `inputs` is always `len(1)`` after `prepare_projection`
def rewrite_elementwise_over(child: ExprIR, /) -> ExprIR:
    """Requested in [discord-0].

    Before:

        nw.col("a").sum().abs().over("b")

    After:

        nw.col("a").sum().over("b").abs()

    [discord-0]: https://discord.com/channels/1235257048170762310/1383078215303696544/1384807793512677398
    """
    if (
        is_window_expr(child)
        and is_function_expr(child.expr)
        and child.expr.options.is_elementwise()
    ):
        # NOTE: Aliasing isn't required, but it does help readability
        window = child
        func = child.expr
        if len(func.input) != 1:
            msg = (
                f"Expected function inputs to have been expanded, "
                f"but got {len(func.input)!r} inputs at: {func}"
            )
            raise NotImplementedError(msg)
        return func.with_input([window.with_expr(func.input[0])])
    return child


# TODO @dangotbanned: Full implementation
def rewrite_binary_agg_over(child: ExprIR, /) -> ExprIR:
    """Requested in [discord-1], clarified in [discord-2].

    Before:

        (nw.col("a") - nw.col("a").mean()).over("b")

    After:

        nw.col("a") - nw.col("a").mean().over("b")

    [discord-1]: https://discord.com/channels/1235257048170762310/1383078215303696544/1384850753008435372
    [discord-2]: https://discord.com/channels/1235257048170762310/1383078215303696544/1384869107203047588
    """
    raise NotImplementedError

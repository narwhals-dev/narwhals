from __future__ import annotations

try:
    import altair as alt
except ImportError as err:
    msg = "`altair` is required to convert `ExprIR`s to transformations."
    raise ModuleNotFoundError(msg) from err

from typing import TYPE_CHECKING, Any, cast

import narwhals._plan as nw
from narwhals._plan.altair.api.expression import into_vega_expr
from narwhals._plan.altair.api.parse import parse_into_named_exprs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.altair.api.typing import IntoAltExpr
    from narwhals._plan.expr import Expr as NwExpr


def calculate_transform(
    *exprs: nw.Expr, **named_exprs: nw.Expr | IntoAltExpr
) -> Iterator[alt.CalculateTransform]:
    if native := tuple(
        name for name, e in named_exprs.items() if not isinstance(e, nw.Expr)
    ):
        for alias in native:
            kwds: dict[str, Any] = {"as": alias, "calculate": named_exprs.pop(alias)}
            yield alt.CalculateTransform(**kwds)
    only_nw = cast("dict[str, NwExpr]", named_exprs)
    for alias, e in parse_into_named_exprs(*exprs, **only_nw):
        yield alt.CalculateTransform(**{"as": alias, "calculate": into_vega_expr(e)})

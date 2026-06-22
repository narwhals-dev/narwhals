from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.altair.expression import into_vega_expr
from narwhals._plan.altair.parse import parse_into_named_exprs

if TYPE_CHECKING:
    from narwhals._plan.altair.parse import IntoExpr
try:
    import altair as alt
except ImportError as err:
    msg = "`altair` is required to convert `ExprIR`s to transformations."
    raise ModuleNotFoundError(msg) from err


def calculate_transform(
    *exprs: IntoExpr, **named_exprs: IntoExpr
) -> list[alt.CalculateTransform]:
    # one transform per expression
    parsed = parse_into_named_exprs(*exprs, **named_exprs)
    return [
        alt.CalculateTransform(**{"as": alias, "calculate": into_vega_expr(e)})
        for alias, e in parsed
    ]

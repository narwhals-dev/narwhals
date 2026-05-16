from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan import _guards, expressions as ir
from narwhals._plan.exceptions import literal_type_error

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import NativeSeriesT
    from narwhals.typing import IntoDType, PythonLiteral


def lit(
    value: PythonLiteral | Series[NativeSeriesT], dtype: IntoDType | None = None
) -> Expr:
    """TODO @dangotbanned: Update the reprs (see `Lit`, `LitSeries`)."""
    if _guards.is_series(value):
        return ir.lit_series(value).to_narwhals()
    if not _guards.is_python_literal(value):
        raise literal_type_error(value)
    return ir.lit(value, dtype).to_narwhals()

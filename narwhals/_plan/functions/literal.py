from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan import _guards, expressions as ir
from narwhals._plan.exceptions import list_literal_error

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import NativeSeriesT
    from narwhals.typing import IntoDType, NonNestedLiteral


def lit(
    value: NonNestedLiteral | Series[NativeSeriesT], dtype: IntoDType | None = None
) -> Expr:
    if _guards.is_series(value):
        return ir.lit_series(value).to_narwhals()
    if not _guards.is_non_nested_literal(value):
        raise list_literal_error(value)
    return ir.lit(value, dtype).to_narwhals()

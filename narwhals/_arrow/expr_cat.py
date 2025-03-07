from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.expr import ArrowExpr


class ArrowExprCatNamespace:
    def __init__(self: Self, expr: ArrowExpr) -> None:
        self._compliant_expr = expr

    def get_categories(self: Self) -> ArrowExpr:
        return self._compliant_expr._reuse_series_namespace("cat", "get_categories")

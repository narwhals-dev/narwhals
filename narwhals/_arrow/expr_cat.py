from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._arrow.utils import ArrowExprNamespace

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.expr import ArrowExpr


class ArrowExprCatNamespace(ArrowExprNamespace):
    def get_categories(self: Self) -> ArrowExpr:
        return self.compliant._reuse_series_namespace("cat", "get_categories")

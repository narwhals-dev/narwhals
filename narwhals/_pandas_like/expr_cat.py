from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.expr import PandasLikeExpr


class PandasLikeExprCatNamespace:
    def __init__(self: Self, expr: PandasLikeExpr) -> None:
        self._compliant_expr = expr

    def get_categories(self: Self) -> PandasLikeExpr:
        return self._compliant_expr._reuse_series_namespace_implementation(
            "cat", "get_categories"
        )

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprListNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def len(self) -> SparkLikeExpr:
        return self._compliant_expr._with_callable(self._compliant_expr._F.array_size)

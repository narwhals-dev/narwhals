from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import ListNamespace

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprListNamespace(
    LazyExprNamespace["SparkLikeExpr"], ListNamespace["SparkLikeExpr"]
):
    def len(self) -> SparkLikeExpr:
        return self.compliant._with_elementwise(self.compliant._F.array_size)

    def unique(self) -> SparkLikeExpr:
        return self.compliant._with_elementwise(self.compliant._F.array_distinct)

    def contains(self, item: Any) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            F = self.compliant._F  # noqa: N806
            return F.array_contains(expr, item)

        return self.compliant._with_elementwise(func)

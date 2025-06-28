from __future__ import annotations

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._compliant.expr import LazyExprNamespace
from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprListNamespace(
    LazyExprNamespace[SparkLikeExpr], ListNamespace[SparkLikeExpr]
):
    def len(self) -> SparkLikeExpr:
        return self.compliant._with_callable(self.compliant._F.array_size)

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._compliant.expr import LazyExprNamespace

if TYPE_CHECKING:
    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprListNamespace(
    LazyExprNamespace["SparkLikeExpr"], ListNamespace["SparkLikeExpr"]
):
    def len(self) -> SparkLikeExpr:
        return self.compliant._with_callable(self.compliant._F.array_size)

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import MapNamespace

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals.typing import NonNestedLiteral


class SparkLikeExprMapNamespace(
    LazyExprNamespace["SparkLikeExpr"], MapNamespace["SparkLikeExpr"]
):
    def get(self, key: NonNestedLiteral) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            # SQLFrame compiles `getItem` with an integer as a 0-based array index.
            F = self.compliant._F
            return F.element_at(expr, F.lit(key))

        return self.compliant._with_elementwise(func)

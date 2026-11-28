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
    def len(self) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            F = self.compliant._F
            # SQLFrame 3.22.0 does not expose map_keys in its DuckDB functions module.
            return F.array_size(F.call_function("map_keys", expr)).cast("int")

        return self.compliant._with_elementwise(func)

    def get(self, key: NonNestedLiteral) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            F = self.compliant._F
            return F.element_at(expr, F.lit(key))

        return self.compliant._with_elementwise(func)

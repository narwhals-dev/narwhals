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
            F = self.compliant._F
            # SQLFrame bug: it doesn't pass the Map type to sqlglot, which then shifts integer
            # keys like array indices. `element_at` is right on 1-based dialects (DuckDB) but
            # off by one on 0-based ones (Snowflake). Should work everywhere once SQLFrame is fixed:
            # https://github.com/eakmanrq/sqlframe/issues/714
            return F.element_at(expr, F.lit(key))

        return self.compliant._with_elementwise(func)

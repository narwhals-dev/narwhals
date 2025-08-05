from __future__ import annotations

from typing import TYPE_CHECKING

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
        def func(expr: Column) -> Column:
            F = self.compliant._F  # noqa: N806
            list_distinct = F.array_distinct(expr)
            return F.when(
                F.array_position(expr, F.lit(None)).isNotNull(),
                F.array_append(list_distinct, F.lit(None)),
            ).otherwise(list_distinct)

        return self.compliant._with_elementwise(func)

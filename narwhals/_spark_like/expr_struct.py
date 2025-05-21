from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.expr import SparkLikeExpr


class SparkLikeExprStructNamespace:
    def __init__(self, expr: SparkLikeExpr) -> None:
        self._compliant_expr = expr

    def field(self, name: str) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            return expr.getField(name)

        return self._compliant_expr._with_callable(func).alias(name)

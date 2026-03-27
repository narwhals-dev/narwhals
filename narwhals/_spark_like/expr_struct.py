from __future__ import annotations

from typing import TYPE_CHECKING, cast

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import StructNamespace

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals.dtypes import Struct


class SparkLikeExprStructNamespace(
    LazyExprNamespace["SparkLikeExpr"], StructNamespace["SparkLikeExpr"]
):
    def field(self, name: str) -> SparkLikeExpr:
        def func(expr: Column) -> Column:
            return expr.getField(name)

        return self.compliant._with_elementwise(func).alias(name)

    def unnest(self) -> SparkLikeExpr:  # pragma: no cover
        compliant = self.compliant

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            schema = df.schema
            return [
                native_expr.getField(field.name).alias(field.name)
                for native_expr, name in zip(
                    compliant(df), compliant._evaluate_output_names(df)
                )
                for field in cast("Struct", schema[name]).fields
            ]

        def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
            schema = df.schema
            return [
                field.name
                for name in compliant._evaluate_output_names(df)
                for field in cast("Struct", schema[name]).fields
            ]

        return compliant.__class__(
            func,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            version=compliant._version,
            implementation=compliant._implementation,
        )

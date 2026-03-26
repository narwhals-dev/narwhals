from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import StructNamespace
from narwhals._duckdb.utils import F, lit

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStructNamespace(
    LazyExprNamespace["DuckDBExpr"], StructNamespace["DuckDBExpr"]
):
    def field(self, name: str) -> DuckDBExpr:
        return self.compliant._with_elementwise(
            lambda expr: F("struct_extract", expr, lit(name))
        ).alias(name)

    def unnest(self) -> DuckDBExpr:
        compliant = self.compliant

        def func(df: DuckDBLazyFrame) -> list[Expression]:
            schema = df.schema
            return [
                F("struct_extract", native_expr, lit(field.name)).alias(field.name)
                for native_expr, name in zip(
                    compliant(df), compliant._evaluate_output_names(df)
                )
                for field in schema[name].fields  # pyright: ignore[reportAttributeAccessIssue]
            ]

        def evaluate_output_names(df: DuckDBLazyFrame) -> list[str]:
            schema = df.schema
            return [
                field.name
                for name in compliant._evaluate_output_names(df)
                for field in schema[name].fields  # pyright: ignore[reportAttributeAccessIssue]
            ]

        return compliant.__class__(
            func,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            version=compliant._version,
        )

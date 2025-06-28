from __future__ import annotations

from narwhals._compliant.any_namespace import StructNamespace
from narwhals._compliant.expr import LazyExprNamespace
from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.utils import F, lit


class DuckDBExprStructNamespace(
    LazyExprNamespace[DuckDBExpr], StructNamespace[DuckDBExpr]
):
    def field(self, name: str) -> DuckDBExpr:
        return self.compliant._with_callable(
            lambda expr: F("struct_extract", expr, lit(name))
        ).alias(name)

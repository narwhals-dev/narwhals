from __future__ import annotations

from narwhals._compliant.any_namespace import ListNamespace
from narwhals._compliant.expr import LazyExprNamespace
from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.utils import F


class DuckDBExprListNamespace(LazyExprNamespace[DuckDBExpr], ListNamespace[DuckDBExpr]):
    def len(self) -> DuckDBExpr:
        return self.compliant._with_callable(lambda expr: F("len", expr))

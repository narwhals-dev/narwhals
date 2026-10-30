from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import LazyExprNamespace
from narwhals._compliant.any_namespace import MapNamespace
from narwhals._duckdb.utils import F, lit

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals.typing import NonNestedLiteral


class DuckDBExprMapNamespace(LazyExprNamespace["DuckDBExpr"], MapNamespace["DuckDBExpr"]):
    def get(self, key: NonNestedLiteral) -> DuckDBExpr:
        def func(expr: Expression) -> Expression:
            if self.compliant._backend_version < (1, 2):  # pragma: no cover
                # `map_extract_value` was added in 1.2; `map_extract` returns a list.
                return F("list_extract", F("map_extract", expr, lit(key)), lit(1))
            return F("map_extract_value", expr, lit(key))

        return self.compliant._with_elementwise(func)

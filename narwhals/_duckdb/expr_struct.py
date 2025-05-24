from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

from narwhals._duckdb.utils import lit

if TYPE_CHECKING:
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprStructNamespace:
    def __init__(self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def field(self, name: str) -> DuckDBExpr:
        return self._compliant_expr._with_callable(
            lambda expr: FunctionExpression("struct_extract", expr, lit(name))
        ).alias(name)

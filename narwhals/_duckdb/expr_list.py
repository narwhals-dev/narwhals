from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duckdb.utils import F

if TYPE_CHECKING:
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprListNamespace:
    def __init__(self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def len(self) -> DuckDBExpr:
        return self._compliant_expr._with_callable(lambda expr: F("len", expr))

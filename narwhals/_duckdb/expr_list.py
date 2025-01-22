from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

if TYPE_CHECKING:
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprListNamespace:
    def __init__(self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def len(self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("len", _input),
            "len",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

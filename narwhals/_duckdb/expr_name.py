from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.expr import CompliantExprNameNamespace
from narwhals._compliant.expr import LazyExprNamespace

if TYPE_CHECKING:
    from narwhals._compliant.typing import AliasName
    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprNameNamespace(
    LazyExprNamespace["DuckDBExpr"], CompliantExprNameNamespace["DuckDBExpr"]
):
    def _from_callable(self, func: AliasName, /, *, alias: bool = True) -> DuckDBExpr:
        expr = self.compliant
        return type(expr)(
            call=expr._call,
            evaluate_output_names=expr._evaluate_output_names,
            alias_output_names=self._alias_output_names(func) if alias else None,
            backend_version=expr._backend_version,
            version=expr._version,
        )

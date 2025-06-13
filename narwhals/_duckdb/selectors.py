from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector, LazySelectorNamespace
from narwhals._duckdb.expr import DuckDBExpr

if TYPE_CHECKING:
    from duckdb import Expression  # noqa: F401

    from narwhals._duckdb.dataframe import DuckDBLazyFrame  # noqa: F401


class DuckDBSelectorNamespace(LazySelectorNamespace["DuckDBLazyFrame", "Expression"]):
    @property
    def _selector(self) -> type[DuckDBSelector]:
        return DuckDBSelector


class DuckDBSelector(  # type: ignore[misc]
    CompliantSelector["DuckDBLazyFrame", "Expression"], DuckDBExpr
):
    def _to_expr(self) -> DuckDBExpr:
        return DuckDBExpr(
            self._call,
            self._window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

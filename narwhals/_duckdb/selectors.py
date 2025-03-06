from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector
from narwhals._compliant import LazySelectorNamespace
from narwhals._duckdb.expr import DuckDBExpr

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._compliant import EvalNames
    from narwhals._compliant import EvalSeries
    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals.utils import _FullContext


class DuckDBSelectorNamespace(
    LazySelectorNamespace["DuckDBLazyFrame", "duckdb.Expression"]
):
    def _selector(
        self,
        call: EvalSeries[DuckDBLazyFrame, duckdb.Expression],
        evaluate_output_names: EvalNames[DuckDBLazyFrame],
        /,
    ) -> DuckDBSelector:
        return DuckDBSelector(
            call,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


class DuckDBSelector(  # type: ignore[misc]
    CompliantSelector["DuckDBLazyFrame", "duckdb.Expression"], DuckDBExpr
):
    def _to_expr(self: Self) -> DuckDBExpr:
        return DuckDBExpr(
            self._call,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator

from duckdb import ColumnExpression

from narwhals._duckdb.expr import DuckDBExpr
from narwhals._selectors import CompliantSelector
from narwhals._selectors import LazySelectorNamespace

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._selectors import EvalNames
    from narwhals._selectors import EvalSeries
    from narwhals.utils import _FullContext


class DuckDBSelectorNamespace(
    LazySelectorNamespace["DuckDBLazyFrame", "duckdb.Expression"]  # type: ignore[type-var]
):
    def _iter_columns(self, df: DuckDBLazyFrame) -> Iterator[duckdb.Expression]:
        for col in df.columns:
            yield ColumnExpression(col)

    def _selector(
        self,
        context: _FullContext,
        call: EvalSeries[DuckDBLazyFrame, duckdb.Expression],  # type: ignore[type-var]
        evaluate_output_names: EvalNames[DuckDBLazyFrame],
        /,
    ) -> CompliantSelector[DuckDBLazyFrame, duckdb.Expression]:  # type: ignore[type-var]
        return DuckDBSelector(
            call,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


class DuckDBSelector(  # type: ignore[misc]
    CompliantSelector["DuckDBLazyFrame", "duckdb.Expression"],  # type: ignore[type-var]
    DuckDBExpr,
):
    @property
    def selectors(self) -> DuckDBSelectorNamespace:
        return DuckDBSelectorNamespace(self)

    def __repr__(self: Self) -> str:  # pragma: no cover
        return f"DuckDBSelector(function_name={self._function_name})"

    def _to_expr(self: Self) -> DuckDBExpr:
        return DuckDBExpr(
            self._call,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

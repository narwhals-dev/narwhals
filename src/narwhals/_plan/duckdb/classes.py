from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from narwhals._plan.common import todo
from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan.duckdb.expr import DuckDBExpr
    from narwhals._plan.duckdb.lazyframe import DuckDBEvaluator, DuckDBLazyFrame

Incomplete: TypeAlias = Any


class DuckDBClasses:
    __slots__ = ()
    version: ClassVar[Version] = Version.MAIN

    @property
    def lazyframe(self) -> type[DuckDBLazyFrame]:
        from narwhals._plan.duckdb.lazyframe import DuckDBLazyFrame

        return DuckDBLazyFrame

    @property
    def evaluator(self) -> type[DuckDBEvaluator]:
        from narwhals._plan.duckdb.lazyframe import DuckDBEvaluator

        return DuckDBEvaluator

    @property
    def expr(self) -> type[DuckDBExpr]:
        from narwhals._plan.duckdb.expr import DuckDBExpr

        return DuckDBExpr

    @property
    def scalar(self) -> type[DuckDBExpr]:
        return self.expr

    v1: Incomplete = todo()
    v2: Incomplete = todo()

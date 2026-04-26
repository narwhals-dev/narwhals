from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan.polars import DataFrame, Expr, LazyFrame, PlanEvaluator, Series


class PolarsClasses:
    __slots__ = ()
    version: ClassVar[Version] = Version.MAIN

    @property
    def _dataframe(self) -> type[DataFrame]:
        from narwhals._plan.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame

    @property
    def _lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.polars.lazyframe import PolarsLazyFrame

        return PolarsLazyFrame

    @property
    def _evaluator(self) -> type[PlanEvaluator]:
        from narwhals._plan.polars.lazyframe import PolarsEvaluator

        return PolarsEvaluator

    @property
    def _expr(self) -> type[Expr]:
        from narwhals._plan.polars.expr import PolarsExpr

        return PolarsExpr

    @property
    def _scalar(self) -> type[Expr]:
        return self._expr

    @property
    def _series(self) -> type[Series]:
        from narwhals._plan.polars.series import PolarsSeries

        return PolarsSeries

    def __narwhals_expr_prepare__(self) -> Expr:
        tp = self._expr
        return tp.__new__(tp)


class PolarsHasClassesTest:
    @property
    def __narwhals_classes__(self) -> PolarsClasses:
        return PolarsClasses()


__narwhals_classes__: Final[PolarsClasses] = PolarsClasses()
"""`mypy` is requiring `[PolarsClasses]`, pyright is fine with `Final`."""

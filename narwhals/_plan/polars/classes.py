from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan.polars import (
        DataFrame,
        Expr,
        LazyFrame,
        PlanEvaluator,
        Series,
        v1,
        v2,
    )


class PolarsClasses:
    __slots__ = ()
    version: ClassVar[Version] = Version.MAIN

    @property
    def dataframe(self) -> type[DataFrame]:
        from narwhals._plan.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame

    @property
    def lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.polars.lazyframe import PolarsLazyFrame

        return PolarsLazyFrame

    @property
    def evaluator(self) -> type[PlanEvaluator]:
        from narwhals._plan.polars.lazyframe import PolarsEvaluator

        return PolarsEvaluator

    @property
    def expr(self) -> type[Expr]:
        from narwhals._plan.polars.expr import PolarsExpr

        return PolarsExpr

    @property
    def scalar(self) -> type[Expr]:
        return self.expr

    @property
    def series(self) -> type[Series]:
        from narwhals._plan.polars.series import PolarsSeries

        return PolarsSeries

    def __narwhals_expr_prepare__(self) -> Expr:
        tp = self.expr
        return tp.__new__(tp)

    @property
    def v1(self) -> PolarsClassesV1:
        return PolarsClassesV1()

    @property
    def v2(self) -> PolarsClassesV2:
        return PolarsClassesV2()


class PolarsClassesV1:
    __slots__ = ()
    version: ClassVar[Version] = Version.V1

    @property
    def dataframe(self) -> type[v1.DataFrame]:
        from narwhals._plan.polars.v1 import DataFrame

        return DataFrame

    @property
    def lazyframe(self) -> type[v1.LazyFrame]:
        from narwhals._plan.polars.v1 import LazyFrame

        return LazyFrame

    @property
    def evaluator(self) -> type[v1.PlanEvaluator]:
        from narwhals._plan.polars.v1 import PlanEvaluator

        return PlanEvaluator

    @property
    def expr(self) -> type[v1.Expr]:
        from narwhals._plan.polars.v1 import Expr

        return Expr

    @property
    def scalar(self) -> type[v1.Scalar]:
        from narwhals._plan.polars.v1 import Scalar

        return Scalar

    @property
    def series(self) -> type[v1.Series]:
        from narwhals._plan.polars.v1 import Series

        return Series

    def __narwhals_expr_prepare__(self) -> v1.Expr:
        tp = self.expr
        return tp.__new__(tp)


class PolarsClassesV2:
    __slots__ = ()
    version: ClassVar[Version] = Version.V2

    @property
    def dataframe(self) -> type[v2.DataFrame]:
        from narwhals._plan.polars.v2 import DataFrame

        return DataFrame

    @property
    def lazyframe(self) -> type[v2.LazyFrame]:
        from narwhals._plan.polars.v2 import LazyFrame

        return LazyFrame

    @property
    def evaluator(self) -> type[v2.PlanEvaluator]:
        from narwhals._plan.polars.v2 import PlanEvaluator

        return PlanEvaluator

    @property
    def expr(self) -> type[v2.Expr]:
        from narwhals._plan.polars.v2 import Expr

        return Expr

    @property
    def scalar(self) -> type[v2.Scalar]:
        from narwhals._plan.polars.v2 import Scalar

        return Scalar

    @property
    def series(self) -> type[v2.Series]:
        from narwhals._plan.polars.v2 import Series

        return Series

    def __narwhals_expr_prepare__(self) -> v2.Expr:
        tp = self.expr
        return tp.__new__(tp)


__narwhals_classes__: Final[PolarsClasses] = PolarsClasses()
"""`mypy` is requiring `[PolarsClasses]`, pyright is fine with `Final`."""

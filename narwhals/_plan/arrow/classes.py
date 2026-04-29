from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan.arrow import DataFrame, Expr, LazyFrame, Scalar, Series, v1, v2


class ArrowClasses:
    __slots__ = ()
    version: ClassVar[Version] = Version.MAIN

    @property
    def _dataframe(self) -> type[DataFrame]:
        from narwhals._plan.arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame

    @property
    def _lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.arrow.lazyframe import ArrowLazyFrame

        return ArrowLazyFrame

    @property
    def _expr(self) -> type[Expr]:
        from narwhals._plan.arrow.expr import ArrowExpr

        return ArrowExpr

    @property
    def _scalar(self) -> type[Scalar]:
        from narwhals._plan.arrow.expr import ArrowScalar

        return ArrowScalar

    @property
    def _series(self) -> type[Series]:
        from narwhals._plan.arrow.series import ArrowSeries

        return ArrowSeries

    def __narwhals_expr_prepare__(self) -> Expr:
        tp = self._expr
        return tp.__new__(tp)

    @property
    def v1(self) -> ArrowClassesV1:
        return ArrowClassesV1()

    @property
    def v2(self) -> ArrowClassesV2:
        return ArrowClassesV2()


class ArrowClassesV1:
    __slots__ = ()
    version: ClassVar[Version] = Version.V1

    @property
    def _dataframe(self) -> type[v1.DataFrame]:
        from narwhals._plan.arrow.v1 import DataFrame

        return DataFrame

    @property
    def _lazyframe(self) -> type[v1.LazyFrame]:
        from narwhals._plan.arrow.v1 import LazyFrame

        return LazyFrame

    @property
    def _expr(self) -> type[v1.Expr]:
        from narwhals._plan.arrow.v1 import Expr

        return Expr

    @property
    def _scalar(self) -> type[v1.Scalar]:
        from narwhals._plan.arrow.v1 import Scalar

        return Scalar

    @property
    def _series(self) -> type[v1.Series]:
        from narwhals._plan.arrow.v1 import Series

        return Series

    def __narwhals_expr_prepare__(self) -> v1.Expr:
        tp = self._expr
        return tp.__new__(tp)


class ArrowClassesV2:
    __slots__ = ()
    version: ClassVar[Version] = Version.V2

    @property
    def _dataframe(self) -> type[v2.DataFrame]:
        from narwhals._plan.arrow.v2 import DataFrame

        return DataFrame

    @property
    def _lazyframe(self) -> type[v2.LazyFrame]:
        from narwhals._plan.arrow.v2 import LazyFrame

        return LazyFrame

    @property
    def _expr(self) -> type[v2.Expr]:
        from narwhals._plan.arrow.v2 import Expr

        return Expr

    @property
    def _scalar(self) -> type[v2.Scalar]:
        from narwhals._plan.arrow.v2 import Scalar

        return Scalar

    @property
    def _series(self) -> type[v2.Series]:
        from narwhals._plan.arrow.v2 import Series

        return Series

    def __narwhals_expr_prepare__(self) -> v2.Expr:
        tp = self._expr
        return tp.__new__(tp)

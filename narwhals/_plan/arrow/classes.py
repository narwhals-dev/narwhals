from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._utils import Version

if TYPE_CHECKING:
    from narwhals._plan.arrow import DataFrame, Expr, LazyFrame, Scalar, Series


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

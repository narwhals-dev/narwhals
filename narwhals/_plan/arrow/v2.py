from __future__ import annotations

from typing import ClassVar, final

from narwhals._plan.arrow.dataframe import ArrowDataFrame
from narwhals._plan.arrow.expr import ArrowExpr, ArrowScalar
from narwhals._plan.arrow.lazyframe import ArrowLazyFrame
from narwhals._plan.arrow.namespace import ArrowNamespace
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
@final
class ArrowExprV2(ArrowExpr):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> ArrowNamespaceV2:
        return ArrowNamespaceV2()


@final
class ArrowScalarV2(ArrowScalar):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> ArrowNamespaceV2:
        return ArrowNamespaceV2()


@final
class ArrowSeriesV2(ArrowSeries):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> ArrowNamespaceV2:
        return ArrowNamespaceV2()


@final
class ArrowDataFrameV2(ArrowDataFrame):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> ArrowNamespaceV2:
        return ArrowNamespaceV2()


@final
class ArrowLazyFrameV2(ArrowLazyFrame):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> ArrowNamespaceV2:
        return ArrowNamespaceV2()


@final
class ArrowNamespaceV2(ArrowNamespace):
    __slots__ = ()
    version: ClassVar = Version.V2

    @property
    def _expr(self) -> type[ArrowExpr]:
        return ArrowExprV2

    @property
    def _scalar(self) -> type[ArrowScalar]:
        return ArrowScalarV2

    @property
    def _series(self) -> type[ArrowSeries]:
        return ArrowSeriesV2

    @property
    def _dataframe(self) -> type[ArrowDataFrameV2]:
        return ArrowDataFrameV2

    @property
    def _lazyframe(self) -> type[ArrowLazyFrame]:
        return ArrowLazyFrameV2


DataFrame = ArrowDataFrameV2
Expr = ArrowExprV2
LazyFrame = ArrowLazyFrameV2
Namespace = ArrowNamespaceV2
PlanEvaluator = None
PlanResolver = None
Scalar = ArrowScalarV2
Series = ArrowSeriesV2


__all__ = (
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Namespace",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
)

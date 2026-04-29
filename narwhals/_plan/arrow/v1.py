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
class ArrowExprV1(ArrowExpr):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespaceV1:
        return ArrowNamespaceV1()


@final
class ArrowScalarV1(ArrowScalar):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespaceV1:
        return ArrowNamespaceV1()


@final
class ArrowSeriesV1(ArrowSeries):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespaceV1:
        return ArrowNamespaceV1()


@final
class ArrowDataFrameV1(ArrowDataFrame):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespaceV1:
        return ArrowNamespaceV1()


@final
class ArrowLazyFrameV1(ArrowLazyFrame):
    __slots__ = ()
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespaceV1:
        return ArrowNamespaceV1()


@final
class ArrowNamespaceV1(ArrowNamespace):
    version: ClassVar = Version.V1

    @property
    def _expr(self) -> type[ArrowExpr]:
        return ArrowExprV1

    @property
    def _scalar(self) -> type[ArrowScalar]:
        return ArrowScalarV1

    @property
    def _series(self) -> type[ArrowSeries]:
        return ArrowSeriesV1

    @property
    def _dataframe(self) -> type[ArrowDataFrameV1]:
        return ArrowDataFrameV1

    @property
    def _lazyframe(self) -> type[ArrowLazyFrame]:
        return ArrowLazyFrameV1


DataFrame = ArrowDataFrameV1
Expr = ArrowExprV1
LazyFrame = ArrowLazyFrameV1
Namespace = ArrowNamespaceV1
PlanEvaluator = None
PlanResolver = None
Scalar = ArrowScalarV1
Series = ArrowSeriesV1


__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Namespace",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
]

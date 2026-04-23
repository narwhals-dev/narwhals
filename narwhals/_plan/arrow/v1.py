from __future__ import annotations

from narwhals._plan.arrow.dataframe import ArrowDataFrame
from narwhals._plan.arrow.expr import ArrowExpr, ArrowScalar
from narwhals._plan.arrow.namespace import ArrowNamespace
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._utils import Version


class ArrowExprV1(ArrowExpr):
    _version = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespaceV1()


class ArrowScalarV1(ArrowScalar):
    _version = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespaceV1()


class ArrowSeriesV1(ArrowSeries):
    _version = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespaceV1()


class ArrowDataFrameV1(ArrowDataFrame):
    _version = Version.V1

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespaceV1()


class ArrowNamespaceV1(ArrowNamespace):
    _version = Version.V1

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
    def _dataframe(self) -> type[ArrowDataFrame]:
        return ArrowDataFrameV1

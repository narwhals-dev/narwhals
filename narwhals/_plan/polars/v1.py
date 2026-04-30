from __future__ import annotations

from typing import ClassVar

from narwhals._plan.polars.dataframe import PolarsDataFrame
from narwhals._plan.polars.expr import PolarsExpr
from narwhals._plan.polars.lazyframe import PolarsEvaluator, PolarsLazyFrame
from narwhals._plan.polars.namespace import PolarsNamespace
from narwhals._plan.polars.series import PolarsSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
class PolarsExprV1(PolarsExpr):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV1()


class PolarsSeriesV1(PolarsSeries):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV1()


class PolarsDataFrameV1(PolarsDataFrame):
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV1()


class PolarsLazyFrameV1(PolarsLazyFrame):
    __slots__ = ()
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV1()


class PolarsEvaluatorV1(PolarsEvaluator):
    __slots__ = ()
    version: ClassVar = Version.V1

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV1()

    _lazyframe: ClassVar = PolarsLazyFrameV1
    # TODO @dangotbanned: Still need to check out the runtime behavior of this
    to_lazy = _lazyframe.from_native


class PolarsNamespaceV1(PolarsNamespace):
    __slots__ = ()
    version: ClassVar = Version.V1

    # NOTE: `_scalar` defers here already
    @property
    def _expr(self) -> type[PolarsExpr]:
        return PolarsExprV1

    @property
    def _series(self) -> type[PolarsSeries]:
        return PolarsSeriesV1

    @property
    def _dataframe(self) -> type[PolarsDataFrame]:
        return PolarsDataFrameV1

    @property
    def _lazyframe(self) -> type[PolarsLazyFrame]:
        return PolarsLazyFrameV1

    @property
    def _evaluator(self) -> type[PolarsEvaluator]:
        return PolarsEvaluatorV1


DataFrame = PolarsDataFrameV1
Expr = PolarsExprV1
LazyFrame = PolarsLazyFrameV1
Namespace = PolarsNamespaceV1
PlanEvaluator = PolarsEvaluatorV1
PlanResolver = None
Scalar = Expr
Series = PolarsSeriesV1


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

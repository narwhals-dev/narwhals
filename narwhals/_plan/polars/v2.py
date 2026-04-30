from __future__ import annotations

from typing import ClassVar

from narwhals._plan.polars.dataframe import PolarsDataFrame
from narwhals._plan.polars.expr import PolarsExpr
from narwhals._plan.polars.lazyframe import PolarsEvaluator, PolarsLazyFrame
from narwhals._plan.polars.namespace import PolarsNamespace
from narwhals._plan.polars.series import PolarsSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
class PolarsExprV2(PolarsExpr):
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV2()


class PolarsSeriesV2(PolarsSeries):
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV2()


class PolarsDataFrameV2(PolarsDataFrame):
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV2()


class PolarsLazyFrameV2(PolarsLazyFrame):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV2()


class PolarsEvaluatorV2(PolarsEvaluator):
    __slots__ = ()
    version: ClassVar = Version.V2

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespaceV2()

    _lazyframe: ClassVar = PolarsLazyFrameV2
    # TODO @dangotbanned: Still need to check out the runtime behavior of this
    to_lazy = _lazyframe.from_native


class PolarsNamespaceV2(PolarsNamespace):
    __slots__ = ()
    version: ClassVar = Version.V2

    # NOTE: `_scalar` defers here already
    @property
    def _expr(self) -> type[PolarsExpr]:
        return PolarsExprV2

    @property
    def _series(self) -> type[PolarsSeries]:
        return PolarsSeriesV2

    @property
    def _dataframe(self) -> type[PolarsDataFrame]:
        return PolarsDataFrameV2

    @property
    def _lazyframe(self) -> type[PolarsLazyFrame]:
        return PolarsLazyFrameV2

    @property
    def _evaluator(self) -> type[PolarsEvaluator]:
        return PolarsEvaluatorV2


DataFrame = PolarsDataFrameV2
Expr = PolarsExprV2
LazyFrame = PolarsLazyFrameV2
Namespace = PolarsNamespaceV2
PlanEvaluator = PolarsEvaluatorV2
PlanResolver = None
Scalar = Expr
Series = PolarsSeriesV2


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

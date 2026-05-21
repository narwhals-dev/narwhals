from __future__ import annotations

from typing import final

from narwhals._plan.arrow.dataframe import ArrowDataFrame
from narwhals._plan.arrow.expr import ArrowExpr, ArrowScalar
from narwhals._plan.arrow.lazyframe import ArrowLazyFrame
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
@final
class ArrowExprV2(ArrowExpr, version=Version.V2):
    __slots__ = ()


@final
class ArrowScalarV2(ArrowScalar, version=Version.V2):
    __slots__ = ()


@final
class ArrowSeriesV2(ArrowSeries, version=Version.V2):
    __slots__ = ()


@final
class ArrowDataFrameV2(ArrowDataFrame, version=Version.V2):
    __slots__ = ()


@final
class ArrowLazyFrameV2(ArrowLazyFrame, version=Version.V2):
    __slots__ = ()


DataFrame = ArrowDataFrameV2
Expr = ArrowExprV2
LazyFrame = ArrowLazyFrameV2
PlanEvaluator = None
PlanResolver = None
Scalar = ArrowScalarV2
Series = ArrowSeriesV2


__all__ = (
    "DataFrame",
    "Expr",
    "LazyFrame",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
)

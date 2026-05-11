from __future__ import annotations

from typing import final

from narwhals._plan.arrow.dataframe import ArrowDataFrame
from narwhals._plan.arrow.expr import ArrowExpr, ArrowScalar
from narwhals._plan.arrow.lazyframe import ArrowLazyFrame
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
@final
class ArrowExprV1(ArrowExpr, version=Version.V1):
    __slots__ = ()


@final
class ArrowScalarV1(ArrowScalar, version=Version.V1):
    __slots__ = ()


@final
class ArrowSeriesV1(ArrowSeries, version=Version.V1):
    __slots__ = ()


@final
class ArrowDataFrameV1(ArrowDataFrame, version=Version.V1):
    __slots__ = ()


@final
class ArrowLazyFrameV1(ArrowLazyFrame, version=Version.V1):
    __slots__ = ()


DataFrame = ArrowDataFrameV1
Expr = ArrowExprV1
LazyFrame = ArrowLazyFrameV1
PlanEvaluator = None
PlanResolver = None
Scalar = ArrowScalarV1
Series = ArrowSeriesV1


__all__ = (
    "DataFrame",
    "Expr",
    "LazyFrame",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
)

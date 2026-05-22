from __future__ import annotations

from typing import ClassVar

from narwhals._plan.polars.classes import PolarsClassesV1
from narwhals._plan.polars.dataframe import PolarsDataFrame
from narwhals._plan.polars.expr import PolarsExpr
from narwhals._plan.polars.lazyframe import PolarsEvaluator, PolarsLazyFrame
from narwhals._plan.polars.series import PolarsSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
class PolarsExprV1(PolarsExpr):
    __slots__ = ()
    version: ClassVar = Version.V1

    @property
    def __narwhals_classes__(self) -> PolarsClassesV1:  # type: ignore[override]
        return PolarsClassesV1()


class PolarsSeriesV1(PolarsSeries):
    __slots__ = ()
    version: ClassVar = Version.V1

    @property
    def __narwhals_classes__(self) -> PolarsClassesV1:  # type: ignore[override]
        return PolarsClassesV1()


class PolarsDataFrameV1(PolarsDataFrame):
    __slots__ = ()
    version: ClassVar = Version.V1

    @property
    def __narwhals_classes__(self) -> PolarsClassesV1:  # type: ignore[override]
        return PolarsClassesV1()


class PolarsLazyFrameV1(PolarsLazyFrame):
    __slots__ = ()
    version: ClassVar = Version.V1

    @property
    def __narwhals_classes__(self) -> PolarsClassesV1:  # type: ignore[override]
        return PolarsClassesV1()


class PolarsEvaluatorV1(PolarsEvaluator):
    __slots__ = ()
    version: ClassVar = Version.V1

    @property
    def __narwhals_classes__(self) -> PolarsClassesV1:  # type: ignore[override]
        return PolarsClassesV1()

    _lazyframe: ClassVar = PolarsLazyFrameV1
    # TODO @dangotbanned: Still need to check out the runtime behavior of this
    to_lazy = _lazyframe.from_native


DataFrame = PolarsDataFrameV1
Expr = PolarsExprV1
LazyFrame = PolarsLazyFrameV1
PlanEvaluator = PolarsEvaluatorV1
PlanResolver = None
Scalar = Expr
Series = PolarsSeriesV1


__all__ = (
    "DataFrame",
    "Expr",
    "LazyFrame",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
)

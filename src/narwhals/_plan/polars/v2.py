from __future__ import annotations

from typing import ClassVar

from narwhals._plan.polars.classes import PolarsClassesV2
from narwhals._plan.polars.dataframe import PolarsDataFrame
from narwhals._plan.polars.expr import PolarsExpr
from narwhals._plan.polars.lazyframe import PolarsEvaluator, PolarsLazyFrame
from narwhals._plan.polars.series import PolarsSeries
from narwhals._utils import Version


# TODO @dangotbanned: Review reducing boilerplate
class PolarsExprV2(PolarsExpr):
    __slots__ = ()
    version: ClassVar = Version.V2

    @property
    def __narwhals_classes__(self) -> PolarsClassesV2:  # type: ignore[override]
        return PolarsClassesV2()


class PolarsSeriesV2(PolarsSeries):
    __slots__ = ()
    version: ClassVar = Version.V2

    @property
    def __narwhals_classes__(self) -> PolarsClassesV2:  # type: ignore[override]
        return PolarsClassesV2()


class PolarsDataFrameV2(PolarsDataFrame):
    __slots__ = ()
    version: ClassVar = Version.V2

    @property
    def __narwhals_classes__(self) -> PolarsClassesV2:  # type: ignore[override]
        return PolarsClassesV2()


class PolarsLazyFrameV2(PolarsLazyFrame):
    __slots__ = ()
    version: ClassVar = Version.V2

    @property
    def __narwhals_classes__(self) -> PolarsClassesV2:  # type: ignore[override]
        return PolarsClassesV2()


class PolarsEvaluatorV2(PolarsEvaluator):
    __slots__ = ()
    version: ClassVar = Version.V2

    @property
    def __narwhals_classes__(self) -> PolarsClassesV2:  # type: ignore[override]
        return PolarsClassesV2()

    _lazyframe: ClassVar = PolarsLazyFrameV2
    # TODO @dangotbanned: Still need to check out the runtime behavior of this
    to_lazy = _lazyframe.from_native


DataFrame = PolarsDataFrameV2
Expr = PolarsExprV2
LazyFrame = PolarsLazyFrameV2
PlanEvaluator = PolarsEvaluatorV2
PlanResolver = None
Scalar = Expr
Series = PolarsSeriesV2


__all__ = (
    "DataFrame",
    "Expr",
    "LazyFrame",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
)

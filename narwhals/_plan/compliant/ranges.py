"""Lazy/eager numeric/temporal range generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import ExprT_co, FrameT_contra
from narwhals._plan.typing import NativeSeriesT, NativeSeriesT_co
from narwhals._utils import Version, _hasattr_static

if TYPE_CHECKING:
    import datetime as dt

    from typing_extensions import TypeIs

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.expressions import ranges
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval

__all__ = (
    "DateRange",
    "DateRangeEager",
    "EagerRangeGenerator",
    "HybridRangeGenerator",
    "IntRange",
    "IntRangeEager",
    "LazyRangeGenerator",
    "LinearSpace",
    "LinearSpaceEager",
    "can_date_range_eager",
    "can_int_range_eager",
    "can_linear_space_eager",
)

Int64 = Version.MAIN.dtypes.Int64()


class DateRange(Protocol[FrameT_contra, ExprT_co]):
    def date_range(
        self, node: ir.RangeExpr[ranges.DateRange], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class IntRange(Protocol[FrameT_contra, ExprT_co]):
    def int_range(
        self, node: ir.RangeExpr[ranges.IntRange], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class LinearSpace(Protocol[FrameT_contra, ExprT_co]):
    def linear_space(
        self, node: ir.RangeExpr[ranges.LinearSpace], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class DateRangeEager(Protocol[NativeSeriesT_co]):
    def date_range_eager(
        self,
        start: dt.date,
        end: dt.date,
        interval: int = 1,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> CompliantSeries[NativeSeriesT_co]: ...


class IntRangeEager(Protocol[NativeSeriesT_co]):
    def int_range_eager(
        self,
        start: int,
        end: int,
        step: int = 1,
        *,
        dtype: IntegerType = Int64,
        name: str = "literal",
    ) -> CompliantSeries[NativeSeriesT_co]: ...


class LinearSpaceEager(Protocol[NativeSeriesT_co]):
    def linear_space_eager(
        self,
        start: float,
        end: float,
        num_samples: int,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> CompliantSeries[NativeSeriesT_co]: ...


class LazyRangeGenerator(
    DateRange[FrameT_contra, ExprT_co],
    IntRange[FrameT_contra, ExprT_co],
    LinearSpace[FrameT_contra, ExprT_co],
    Protocol[FrameT_contra, ExprT_co],
):
    """Supports all range generation methods that return expressions.

    `[FrameT_contra, ExprT_co]`.
    """


class EagerRangeGenerator(
    DateRangeEager[NativeSeriesT_co],
    IntRangeEager[NativeSeriesT_co],
    LinearSpaceEager[NativeSeriesT_co],
    Protocol[NativeSeriesT_co],
):
    """Supports all range generation methods that return series.

    `[NativeSeriesT_co]`.
    """


class HybridRangeGenerator(
    LazyRangeGenerator[FrameT_contra, ExprT_co],
    EagerRangeGenerator[NativeSeriesT_co],
    Protocol[FrameT_contra, ExprT_co, NativeSeriesT_co],
):
    """Supports all range generation methods.

    `[FrameT_contra, ExprT_co, NativeSeriesT_co]`.
    """


# fmt: off
def can_date_range_eager(obj: DateRangeEager[NativeSeriesT] | Any) -> TypeIs[DateRangeEager[NativeSeriesT]]:
    return _hasattr_static(obj, "date_range_eager")
def can_int_range_eager(obj: IntRangeEager[NativeSeriesT] | Any) -> TypeIs[IntRangeEager[NativeSeriesT]]:
    return _hasattr_static(obj, "int_range_eager")
def can_linear_space_eager(obj: LinearSpaceEager[NativeSeriesT] | Any) -> TypeIs[LinearSpaceEager[NativeSeriesT]]:
    return _hasattr_static(obj, "linear_space_eager")
# fmt: on

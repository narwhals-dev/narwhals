"""Lazy/eager numeric/temporal range generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.typing import (
    ExprT,
    ExprT_co,
    FrameT,
    FrameT_contra,
    SeriesT,
    SeriesT_co,
)
from narwhals._utils import Version, _hasattr_static

if TYPE_CHECKING:
    import datetime as dt

    from typing_extensions import TypeIs

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import ranges
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval

# TODO @dangotbanned: Redo `*Namespace` in terms of these
__all__ = [
    "DateRange",
    "DateRangeEager",
    "EagerRangeGenerator",
    "HybridRangeGenerator",
    "IntRange",
    "IntRangeEager",
    "LazyRangeGenerator",
    "LinearSpace",
    "LinearSpaceEager",
    "can_date_range",
    "can_date_range_eager",
    "can_int_range",
    "can_int_range_eager",
    "can_linear_space",
    "can_linear_space_eager",
]

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


class DateRangeEager(Protocol[SeriesT_co]):
    def date_range_eager(
        self,
        start: dt.date,
        end: dt.date,
        interval: int = 1,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> SeriesT_co: ...


class IntRangeEager(Protocol[SeriesT_co]):
    def int_range_eager(
        self,
        start: int,
        end: int,
        step: int = 1,
        *,
        dtype: IntegerType = Int64,
        name: str = "literal",
    ) -> SeriesT_co: ...


class LinearSpaceEager(Protocol[SeriesT_co]):
    def linear_space_eager(
        self,
        start: float,
        end: float,
        num_samples: int,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> SeriesT_co: ...


class LazyRangeGenerator(
    DateRange[FrameT_contra, ExprT_co],
    IntRange[FrameT_contra, ExprT_co],
    LinearSpace[FrameT_contra, ExprT_co],
    Protocol[FrameT_contra, ExprT_co],
):
    """Supports all range generation methods that return expressions."""


class EagerRangeGenerator(
    DateRangeEager[SeriesT_co],
    IntRangeEager[SeriesT_co],
    LinearSpaceEager[SeriesT_co],
    Protocol[SeriesT_co],
):
    """Supports all range generation methods that return series."""


class HybridRangeGenerator(
    LazyRangeGenerator[FrameT_contra, ExprT_co], EagerRangeGenerator[SeriesT_co]
):
    """Supports all range generation methods.."""


# TODO @dangotbanned: Use these in https://github.com/narwhals-dev/narwhals/blob/40bf84f1f226f29e86b8ba8534f87a026cdf62ae/narwhals/_plan/functions/ranges.py
# fmt: off
def can_date_range(obj: DateRange[FrameT, ExprT] | Any) -> TypeIs[DateRange[FrameT, ExprT]]:
    return _hasattr_static(obj, "date_range")
def can_int_range(obj: IntRange[FrameT, ExprT] | Any) -> TypeIs[IntRange[FrameT, ExprT]]:
    return _hasattr_static(obj, "int_range")
def can_linear_space(obj: LinearSpace[FrameT, ExprT] | Any) -> TypeIs[LinearSpace[FrameT, ExprT]]:
    return _hasattr_static(obj, "linear_space")
def can_date_range_eager(obj: DateRangeEager[SeriesT] | Any) -> TypeIs[DateRangeEager[SeriesT]]:
    return _hasattr_static(obj, "date_range_eager")
def can_int_range_eager(obj: IntRangeEager[SeriesT] | Any) -> TypeIs[IntRangeEager[SeriesT]]:
    return _hasattr_static(obj, "int_range_eager")
def can_linear_space_eager(obj: LinearSpaceEager[SeriesT] | Any) -> TypeIs[LinearSpaceEager[SeriesT]]:
    return _hasattr_static(obj, "linear_space_eager")
# fmt: on

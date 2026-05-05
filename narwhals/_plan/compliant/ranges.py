"""Lazy numeric/temporal range generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from narwhals._plan.compliant.typing import ExprT_co, FrameT_contra

if TYPE_CHECKING:
    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import ranges

__all__ = ("DateRange", "IntRange", "LazyRangeGenerator", "LinearSpace")


class DateRange(Protocol[FrameT_contra, ExprT_co]):
    __slots__ = ()

    def date_range(
        self, node: ir.RangeExpr[ranges.DateRange], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class IntRange(Protocol[FrameT_contra, ExprT_co]):
    __slots__ = ()

    def int_range(
        self, node: ir.RangeExpr[ranges.IntRange], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class LinearSpace(Protocol[FrameT_contra, ExprT_co]):
    __slots__ = ()

    def linear_space(
        self, node: ir.RangeExpr[ranges.LinearSpace], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...


class LazyRangeGenerator(
    DateRange[FrameT_contra, ExprT_co],
    IntRange[FrameT_contra, ExprT_co],
    LinearSpace[FrameT_contra, ExprT_co],
    Protocol[FrameT_contra, ExprT_co],
):
    """Supports all range generation methods that return expressions.

    `[FrameT_contra, ExprT_co]`.
    """

    __slots__ = ()

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.compliant.column import EagerBroadcast, SupportsBroadcast
from narwhals._plan.compliant.typing import (
    FrameT_contra,
    HasVersion,
    LengthT,
    SeriesT,
    SeriesT_co,
)
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.scalar import CompliantScalar
    from narwhals._plan.expressions import (
        BinaryExpr,
        FunctionExpr,
        aggregation as agg,
        boolean,
        functions as F,
    )
    from narwhals._plan.expressions.boolean import (
        IsBetween,
        IsFinite,
        IsFirstDistinct,
        IsLastDistinct,
        IsNan,
        IsNull,
        Not,
    )


class CompliantExpr(HasVersion, Protocol[FrameT_contra, SeriesT_co]):
    """Everything common to `Expr`/`Series` and `Scalar` literal values."""

    _evaluated: Any
    """Compliant or native value."""

    def _with_native(self, native: Any, name: str, /) -> Self:
        return self.from_native(native, name or self.name, self.version)

    @classmethod
    def from_native(
        cls, native: Any, name: str = "", /, version: Version = Version.MAIN
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    # series & scalar
    def abs(self, node: FunctionExpr[F.Abs], frame: FrameT_contra, name: str) -> Self: ...
    def binary_expr(self, node: BinaryExpr, frame: FrameT_contra, name: str) -> Self: ...
    def cast(self, node: ir.Cast, frame: FrameT_contra, name: str) -> Self: ...
    def ewm_mean(
        self, node: FunctionExpr[F.EwmMean], frame: FrameT_contra, name: str
    ) -> Self: ...
    def fill_null(
        self, node: FunctionExpr[F.FillNull], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_between(
        self, node: FunctionExpr[IsBetween], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_finite(
        self, node: FunctionExpr[IsFinite], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_first_distinct(
        self, node: FunctionExpr[IsFirstDistinct], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_last_distinct(
        self, node: FunctionExpr[IsLastDistinct], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_nan(
        self, node: FunctionExpr[IsNan], frame: FrameT_contra, name: str
    ) -> Self: ...
    def is_null(
        self, node: FunctionExpr[IsNull], frame: FrameT_contra, name: str
    ) -> Self: ...
    def map_batches(
        self, node: ir.AnonymousExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def not_(self, node: FunctionExpr[Not], frame: FrameT_contra, name: str) -> Self: ...
    def over(self, node: ir.WindowExpr, frame: FrameT_contra, name: str) -> Self: ...
    # NOTE: `Scalar` is returned **only** for un-partitioned `OrderableAggExpr`
    # e.g. `nw.col("a").first().over(order_by="b")`
    def over_ordered(
        self, node: ir.OrderedWindowExpr, frame: FrameT_contra, name: str
    ) -> Self | CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def pow(self, node: FunctionExpr[F.Pow], frame: FrameT_contra, name: str) -> Self: ...
    def rolling_expr(
        self, node: ir.RollingExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    def shift(
        self, node: FunctionExpr[F.Shift], frame: FrameT_contra, name: str
    ) -> Self: ...
    def ternary_expr(
        self, node: ir.TernaryExpr, frame: FrameT_contra, name: str
    ) -> Self: ...
    # series only
    def filter(self, node: ir.Filter, frame: FrameT_contra, name: str) -> Self: ...
    def sort(self, node: ir.Sort, frame: FrameT_contra, name: str) -> Self: ...
    def sort_by(self, node: ir.SortBy, frame: FrameT_contra, name: str) -> Self: ...
    def diff(
        self, node: FunctionExpr[F.Diff], frame: FrameT_contra, name: str
    ) -> Self: ...
    def cum_sum(
        self, node: FunctionExpr[F.CumSum], frame: FrameT_contra, name: str
    ) -> Self: ...
    # series -> scalar
    def all(
        self, node: FunctionExpr[boolean.All], frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def any(
        self, node: FunctionExpr[boolean.Any], frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def arg_max(
        self, node: agg.ArgMax, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def arg_min(
        self, node: agg.ArgMin, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def count(
        self, node: agg.Count, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def first(
        self, node: agg.First, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def last(
        self, node: agg.Last, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def len(
        self, node: agg.Len, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def max(
        self, node: agg.Max, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def mean(
        self, node: agg.Mean, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def median(
        self, node: agg.Median, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def min(
        self, node: agg.Min, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def n_unique(
        self, node: agg.NUnique, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def quantile(
        self, node: agg.Quantile, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def sum(
        self, node: agg.Sum, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def std(
        self, node: agg.Std, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...
    def var(
        self, node: agg.Var, frame: FrameT_contra, name: str
    ) -> CompliantScalar[FrameT_contra, SeriesT_co]: ...


class EagerExpr(
    EagerBroadcast[SeriesT],
    CompliantExpr[FrameT_contra, SeriesT],
    Protocol[FrameT_contra, SeriesT],
): ...


class LazyExpr(
    SupportsBroadcast[SeriesT, LengthT],
    CompliantExpr[FrameT_contra, SeriesT],
    Protocol[FrameT_contra, SeriesT, LengthT],
): ...

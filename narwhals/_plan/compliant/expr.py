from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from narwhals._plan.compliant.column import EagerBroadcast
from narwhals._plan.compliant.typing import FrameT_contra, HasVersion, SeriesT

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.accessors import (
        ExprCatNamespace,
        ExprDateTimeNamespace,
        ExprListNamespace,
        ExprStringNamespace,
        ExprStructNamespace,
    )
    from narwhals._plan.compliant.scalar import CompliantScalar, EagerScalar
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
        IsNotNan,
        IsNotNull,
        IsNull,
        Not,
    )
    from narwhals._plan.typing import IncompleteCyclic


# TODO @dangotbanned: Rethink `FrameT_contra`
# - Doesn't need a bound at this level
# - Has to coordinate the change with these, but none actually require a `CompliantFrame`:
#   - Needs `frame.version`
#     - `compliant.column.ExprDispatch` (`compliant.typing.Ctx`)
#   - Needs `ExprDispatch`
#     - `ExprIR.dispatch`
#     - `_dispatch.{Binder,BoundMethod}`
#     - `_dispatch.Dispatcher.{bind,__call__}`
class CompliantExpr(HasVersion, Protocol[FrameT_contra]):
    """Everything common to `Expr`/`Series` and `Scalar` literal values.

    `[FrameT_contra]`.
    """

    # Expr -> Expr
    # Scalar -> Scalar
    def abs(
        self, node: FunctionExpr[F.Abs], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def binary_expr(
        self, node: BinaryExpr, frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def cast(self, node: ir.Cast, frame: FrameT_contra, name: str, /) -> Self: ...
    def ceil(
        self, node: FunctionExpr[F.Ceil], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def clip(
        self, node: FunctionExpr[F.Clip], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def clip_lower(
        self, node: FunctionExpr[F.ClipLower], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def clip_upper(
        self, node: FunctionExpr[F.ClipUpper], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def drop_nulls(
        self, node: FunctionExpr[F.DropNulls], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def ewm_mean(
        self, node: FunctionExpr[F.EwmMean], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def exp(
        self, node: FunctionExpr[F.Exp], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def fill_null(
        self, node: FunctionExpr[F.FillNull], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def fill_null_with_strategy(
        self,
        node: FunctionExpr[F.FillNullWithStrategy],
        frame: FrameT_contra,
        name: str,
        /,
    ) -> Self: ...
    def floor(
        self, node: FunctionExpr[F.Floor], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def hist_bins(
        self, node: FunctionExpr[F.HistBins], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def hist_bin_count(
        self, node: FunctionExpr[F.HistBinCount], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_between(
        self, node: FunctionExpr[IsBetween], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_duplicated(
        self, node: FunctionExpr[boolean.IsDuplicated], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_finite(
        self, node: FunctionExpr[IsFinite], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_first_distinct(
        self, node: FunctionExpr[IsFirstDistinct], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_in_expr(
        self, node: FunctionExpr[boolean.IsInExpr], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_in_seq(
        self, node: FunctionExpr[boolean.IsInSeq], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_last_distinct(
        self, node: FunctionExpr[IsLastDistinct], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_nan(
        self, node: FunctionExpr[IsNan], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_null(
        self, node: FunctionExpr[IsNull], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_not_nan(
        self, node: FunctionExpr[IsNotNan], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_not_null(
        self, node: FunctionExpr[IsNotNull], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_unique(
        self, node: FunctionExpr[boolean.IsUnique], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def log(
        self, node: FunctionExpr[F.Log], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def mode_all(
        self, node: FunctionExpr[F.ModeAll], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def not_(
        self, node: FunctionExpr[Not], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def over(self, node: ir.Over, frame: FrameT_contra, name: str, /) -> Self: ...
    def pow(
        self, node: FunctionExpr[F.Pow], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def replace_strict(
        self, node: FunctionExpr[F.ReplaceStrict], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def replace_strict_default(
        self,
        node: FunctionExpr[F.ReplaceStrictDefault],
        frame: FrameT_contra,
        name: str,
        /,
    ) -> Self: ...
    def rolling_mean(
        self, node: FunctionExpr[F.RollingMean], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def rolling_sum(
        self, node: FunctionExpr[F.RollingSum], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def rolling_std(
        self, node: FunctionExpr[F.RollingStd], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def rolling_var(
        self, node: FunctionExpr[F.RollingVar], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def round(
        self, node: FunctionExpr[F.Round], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def shift(
        self, node: FunctionExpr[F.Shift], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def sqrt(
        self, node: FunctionExpr[F.Sqrt], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def ternary_expr(
        self, node: ir.TernaryExpr, frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def unique(
        self, node: FunctionExpr[F.Unique], frame: FrameT_contra, name: str, /
    ) -> Self: ...

    # Expr -> Expr
    # (some are `Scalar` noops)
    def cum_count(
        self, node: FunctionExpr[F.CumCount], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def cum_max(
        self, node: FunctionExpr[F.CumMax], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def cum_min(
        self, node: FunctionExpr[F.CumMin], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def cum_prod(
        self, node: FunctionExpr[F.CumProd], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def cum_sum(
        self, node: FunctionExpr[F.CumSum], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def diff(
        self, node: FunctionExpr[F.Diff], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def filter(self, node: ir.Filter, frame: FrameT_contra, name: str, /) -> Self: ...
    def rank(
        self, node: FunctionExpr[F.Rank], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def sort(self, node: ir.Sort, frame: FrameT_contra, name: str, /) -> Self: ...
    def sort_by(self, node: ir.SortBy, frame: FrameT_contra, name: str, /) -> Self: ...

    # Expr -> Scalar
    # TODO @dangotbanned: Move this concept to the ExprIR layer?
    # - Every `Function` has `FunctionFlags.AGGREGATION`
    # - Everything* else is an `AggExpr`
    # - `OverOrdered` is an outlier, and it also doesn't specify `is_scalar`?
    def all(
        self, node: FunctionExpr[boolean.All], frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def any(
        self, node: FunctionExpr[boolean.Any], frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def arg_max(
        self, node: agg.ArgMax, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def arg_min(
        self, node: agg.ArgMin, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def count(
        self, node: agg.Count, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def first(
        self, node: agg.First, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def last(
        self, node: agg.Last, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def len(
        self, node: agg.Len, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def max(
        self, node: agg.Max, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def mean(
        self, node: agg.Mean, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def median(
        self, node: agg.Median, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def min(
        self, node: agg.Min, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def mode_any(
        self, node: FunctionExpr[F.ModeAny], frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def n_unique(
        self, node: agg.NUnique, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def null_count(
        self, node: FunctionExpr[F.NullCount], frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    # NOTE: `Scalar` is returned **only** for un-partitioned `OrderableAggExpr`
    #  - e.g. `nw.col("a").first().over(order_by="b")`
    # TODO @dangotbanned: Split (un-partitioned + ordered) into another node?
    # - The handling of the union would need repeating everything otherwise
    # - https://github.com/narwhals-dev/narwhals/blob/489bada3e9318f91c9d73744e7a6de62d2478451/narwhals/_plan/arrow/expr.py#L560-L570
    def over_ordered(
        self, node: ir.OverOrdered, frame: FrameT_contra, name: str, /
    ) -> Self | CompliantScalar[FrameT_contra]: ...
    def quantile(
        self, node: agg.Quantile, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def sum(
        self, node: agg.Sum, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def std(
        self, node: agg.Std, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def var(
        self, node: agg.Var, frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def kurtosis(
        self, node: FunctionExpr[F.Kurtosis], frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...
    def skew(
        self, node: FunctionExpr[F.Skew], frame: FrameT_contra, name: str, /
    ) -> CompliantScalar[FrameT_contra]: ...

    @property
    def cat(self) -> ExprCatNamespace[FrameT_contra, CompliantExpr[FrameT_contra]]: ...
    @property
    def dt(
        self,
    ) -> ExprDateTimeNamespace[FrameT_contra, CompliantExpr[FrameT_contra]]: ...
    @property
    def list(self) -> ExprListNamespace[FrameT_contra, CompliantExpr[FrameT_contra]]: ...
    @property
    def str(self) -> ExprStringNamespace[FrameT_contra, CompliantExpr[FrameT_contra]]: ...
    @property
    def struct(
        self,
    ) -> ExprStructNamespace[FrameT_contra, CompliantExpr[FrameT_contra]]: ...


class EagerExpr(
    EagerBroadcast[SeriesT],
    CompliantExpr[FrameT_contra],
    Protocol[FrameT_contra, SeriesT],
):
    """`[FrameT_contra, SeriesT]`."""

    def __bool__(self) -> Literal[True]:
        # NOTE: Avoids falling back to `__len__` (via `EagerBroadcast`) when truth-testing on dispatch
        return True

    def gather_every(
        self, node: FunctionExpr[F.GatherEvery], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    def is_in_series(
        self,
        node: FunctionExpr[boolean.IsInSeries[IncompleteCyclic]],
        frame: FrameT_contra,
        name: str,
        /,
    ) -> Self: ...
    # NOTE: `Scalar` when using `returns_scalar=True`
    def map_batches(
        self, node: ir.AnonymousExpr, frame: FrameT_contra, name: str, /
    ) -> Self | EagerScalar[FrameT_contra, SeriesT]: ...
    def sample_frac(
        self, node: FunctionExpr[F.SampleFrac], frame: FrameT_contra, name: str, /
    ) -> Self: ...
    # NOTE: `n=1` can behave similar to an aggregation in `select(...)`, but requires `.first()`
    # to trigger broadcasting in `with_columns(...)`
    def sample_n(
        self, node: FunctionExpr[F.SampleN], frame: FrameT_contra, name: str, /
    ) -> Self: ...

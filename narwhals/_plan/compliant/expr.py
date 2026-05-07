from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar

from narwhals._plan.compliant.broadcast import BroadcastSeries
from narwhals._plan.compliant.typing import (
    EagerDataFrameT_contra as EagerFrame,
    FrameT_contra as Frame,
    NativeExpr_co,
    NativeScalar_co,
    NativeSeriesT,
)

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.accessors import (
        ExprCatNamespace,
        ExprDateTimeNamespace,
        ExprListNamespace,
        ExprStringNamespace,
        ExprStructNamespace,
    )
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar as Scalar, EagerScalar
    from narwhals._plan.expressions import (
        BinaryExpr,
        FunctionExpr as FExpr,
        HorizontalExpr as HExpr,
        aggregation as agg,
        functions as F,
        strings,
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
    from narwhals._plan.typing import IncompleteCyclic, IncompleteVarianceLie
    from narwhals._utils import Version
    from narwhals.typing import PythonLiteral

Incomplete: TypeAlias = Any


IncompleteDispatch: TypeAlias = Incomplete
"""Placeholder for `DispatchScope`/namespace matching types.

*Previously* this was working around a cycle for:
- `CompliantExpr`
- `CompliantScalar`
- `CompliantNamespace`

The next iteration needs to use `__narwhals_classes__` and
be more careful with where *that* enters the protocols.

Note:
    This ends up being the return type for `FunctionExpr.dispatch_arg(s)`
"""


Native_co = TypeVar("Native_co", covariant=True)


class CompliantColumn(Protocol[Frame, Native_co]):
    __slots__ = ()
    version: ClassVar[Version]

    @property
    def native(self) -> Native_co: ...

    # Constructors (Scalar)  # noqa: ERA001
    @classmethod
    def len_star(
        cls, node: ir.Len, frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    @classmethod
    def lit(
        cls, node: ir.Lit[PythonLiteral], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...

    # NOTE: May need to change returning `Self` to `CompliantColumn[FrameT]`
    # Expr -> Expr
    # Scalar -> Scalar
    def abs(self, node: FExpr[F.Abs], frame: Frame, name: str, /) -> Self: ...
    def binary_expr(self, node: BinaryExpr, frame: Frame, name: str, /) -> Self: ...
    def cast(self, node: ir.Cast, frame: Frame, name: str, /) -> Self: ...
    def ceil(self, node: FExpr[F.Ceil], frame: Frame, name: str, /) -> Self: ...
    def clip(self, node: FExpr[F.Clip], frame: Frame, name: str, /) -> Self: ...
    def clip_lower(
        self, node: FExpr[F.ClipLower], frame: Frame, name: str, /
    ) -> Self: ...
    def clip_upper(
        self, node: FExpr[F.ClipUpper], frame: Frame, name: str, /
    ) -> Self: ...
    def exp(self, node: FExpr[F.Exp], frame: Frame, name: str, /) -> Self: ...
    def fill_null(self, node: FExpr[F.FillNull], frame: Frame, name: str, /) -> Self: ...
    def floor(self, node: FExpr[F.Floor], frame: Frame, name: str, /) -> Self: ...
    def is_between(self, node: FExpr[IsBetween], frame: Frame, name: str, /) -> Self: ...
    def is_finite(self, node: FExpr[IsFinite], frame: Frame, name: str, /) -> Self: ...
    def is_in_expr(
        self, node: FExpr[ir.boolean.IsInExpr], frame: Frame, name: str, /
    ) -> Self: ...
    def is_in_seq(
        self, node: FExpr[ir.boolean.IsInSeq], frame: Frame, name: str, /
    ) -> Self: ...
    def is_nan(self, node: FExpr[IsNan], frame: Frame, name: str, /) -> Self: ...
    def is_null(self, node: FExpr[IsNull], frame: Frame, name: str, /) -> Self: ...
    def is_not_nan(self, node: FExpr[IsNotNan], frame: Frame, name: str, /) -> Self: ...
    def is_not_null(self, node: FExpr[IsNotNull], frame: Frame, name: str, /) -> Self: ...
    def log(self, node: FExpr[F.Log], frame: Frame, name: str, /) -> Self: ...
    def not_(self, node: FExpr[Not], frame: Frame, name: str, /) -> Self: ...
    def pow(self, node: FExpr[F.Pow], frame: Frame, name: str, /) -> Self: ...
    def replace_strict(
        self, node: FExpr[F.ReplaceStrict], frame: Frame, name: str, /
    ) -> Self: ...
    def replace_strict_default(
        self, node: FExpr[F.ReplaceStrictDefault], frame: Frame, name: str, /
    ) -> Self: ...
    def round(self, node: FExpr[F.Round], frame: Frame, name: str, /) -> Self: ...
    def sqrt(self, node: FExpr[F.Sqrt], frame: Frame, name: str, /) -> Self: ...
    def ternary_expr(self, node: ir.TernaryExpr, frame: Frame, name: str, /) -> Self: ...

    # `Scalar` has defined-behavior
    def drop_nulls(
        self, node: FExpr[F.DropNulls], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def ewm_mean(self, node: FExpr[F.EwmMean], frame: Frame, name: str, /) -> Self: ...
    def shift(self, node: FExpr[F.Shift], frame: Frame, name: str, /) -> Self: ...
    def is_duplicated(
        self, node: FExpr[ir.boolean.IsDuplicated], frame: Frame, name: str, /
    ) -> Self: ...
    def is_first_distinct(
        self, node: FExpr[IsFirstDistinct], frame: Frame, name: str, /
    ) -> Self: ...
    def is_last_distinct(
        self, node: FExpr[IsLastDistinct], frame: Frame, name: str, /
    ) -> Self: ...
    def is_unique(
        self, node: FExpr[ir.boolean.IsUnique], frame: Frame, name: str, /
    ) -> Self: ...

    # `Scalar` no-op
    def sort(self, node: ir.Sort, frame: Frame, name: str, /) -> Self: ...
    def sort_by(self, node: ir.SortBy, frame: Frame, name: str, /) -> Self: ...
    def unique(self, node: FExpr[F.Unique], frame: Frame, name: str, /) -> Self: ...

    # (Scalar, ...)              -> Scalar
    # (Expr, ...)                -> Expr
    # (Expr, Expr | Scalar, ...) -> Expr
    def all_horizontal(
        self, node: HExpr[ir.boolean.AllHorizontal], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def any_horizontal(
        self, node: HExpr[ir.boolean.AnyHorizontal], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def coalesce(
        self, node: HExpr[F.Coalesce], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def concat_str(
        self, node: HExpr[strings.ConcatStr], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def max_horizontal(
        self, node: HExpr[F.MaxHorizontal], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def min_horizontal(
        self, node: HExpr[F.MinHorizontal], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...
    def sum_horizontal(
        self, node: HExpr[F.SumHorizontal], frame: Frame, name: str, /
    ) -> Self | CompliantColumn[Frame, Incomplete]: ...

    def __narwhals_namespace__(
        self,
    ) -> CompliantNamespace[
        IncompleteVarianceLie, IncompleteDispatch, IncompleteDispatch
    ]: ...

    def __narwhals_expr_prepare__(self) -> Self:
        """Return a partially initialized instance of this class.

        The only external (narwhals-level) requirement is that we have an instance to call methods on.
        """
        tp = type(self)
        return tp.__new__(tp)

    @property
    def cat(self) -> ExprCatNamespace[Frame, CompliantColumn[Frame, Incomplete]]: ...
    @property
    def dt(self) -> ExprDateTimeNamespace[Frame, CompliantColumn[Frame, Incomplete]]: ...
    @property
    def list(self) -> ExprListNamespace[Frame, CompliantColumn[Frame, Incomplete]]: ...
    @property
    def str(self) -> ExprStringNamespace[Frame, CompliantColumn[Frame, Incomplete]]: ...
    @property
    def struct(
        self,
    ) -> ExprStructNamespace[Frame, CompliantColumn[Frame, Incomplete]]: ...


# TODO @dangotbanned: Avoid `FrameT`?
# TODO @dangotbanned: Binary Namespace methods -> Expr methods
# - [ ] `date_range`
# - [ ] `int_range`
# - [ ] `linear_space`
class CompliantExpr(
    CompliantColumn[Frame, NativeExpr_co], Protocol[Frame, NativeExpr_co, NativeScalar_co]
):
    """Everything common to `Expr` and `Scalar` literal values.

    `[FrameT, NativeExpr_co, NativeScalar_co]`.
    """

    __slots__ = ()

    # Constructor
    @classmethod
    def col(cls, node: ir.Column, frame: Frame, name: str, /) -> Self: ...

    def fill_null_with_strategy(
        self, node: FExpr[F.FillNullWithStrategy], frame: Frame, name: str, /
    ) -> Self: ...
    def mode_all(self, node: FExpr[F.ModeAll], frame: Frame, name: str, /) -> Self: ...

    # has defined behavior for the empty expr case?
    def hist_bins(self, node: FExpr[F.HistBins], frame: Frame, name: str, /) -> Self: ...
    def hist_bin_count(
        self, node: FExpr[F.HistBinCount], frame: Frame, name: str, /
    ) -> Self: ...

    # complicated
    def over(self, node: ir.Over, frame: Frame, name: str, /) -> Self: ...
    # NOTE: `Scalar` is returned **only** for un-partitioned `OrderableAggExpr`
    #  - e.g. `nw.col("a").first().over(order_by="b")`
    # TODO @dangotbanned: Split (un-partitioned + ordered) into another node?
    # - The handling of the union would need repeating everything otherwise
    # - https://github.com/narwhals-dev/narwhals/blob/489bada3e9318f91c9d73744e7a6de62d2478451/narwhals/_plan/arrow/expr.py#L560-L570
    # un-partitioned is a possible no-op for `Scalar` (be careful as a single aggregation doesn't make it scalar)
    def over_ordered(
        self, node: ir.OverOrdered, frame: Frame, name: str, /
    ) -> Self | Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...

    def rolling_mean(
        self, node: FExpr[F.RollingMean], frame: Frame, name: str, /
    ) -> Self: ...
    def rolling_sum(
        self, node: FExpr[F.RollingSum], frame: Frame, name: str, /
    ) -> Self: ...
    def rolling_std(
        self, node: FExpr[F.RollingStd], frame: Frame, name: str, /
    ) -> Self: ...
    def rolling_var(
        self, node: FExpr[F.RollingVar], frame: Frame, name: str, /
    ) -> Self: ...

    def cum_count(self, node: FExpr[F.CumCount], frame: Frame, name: str, /) -> Self: ...
    def cum_max(self, node: FExpr[F.CumMax], frame: Frame, name: str, /) -> Self: ...
    def cum_min(self, node: FExpr[F.CumMin], frame: Frame, name: str, /) -> Self: ...
    def cum_prod(self, node: FExpr[F.CumProd], frame: Frame, name: str, /) -> Self: ...
    def cum_sum(self, node: FExpr[F.CumSum], frame: Frame, name: str, /) -> Self: ...
    def diff(self, node: FExpr[F.Diff], frame: Frame, name: str, /) -> Self: ...
    def filter(self, node: ir.Filter, frame: Frame, name: str, /) -> Self: ...
    def rank(self, node: FExpr[F.Rank], frame: Frame, name: str, /) -> Self: ...

    # Expr -> Scalar
    # TODO @dangotbanned: Move this concept to the ExprIR layer?
    # - Every `Function` has `FunctionFlags.AGGREGATION`
    # - Everything* else is an `AggExpr`
    # - `OverOrdered` is an outlier, and it also doesn't specify `is_scalar`?
    def all(
        self, node: FExpr[ir.boolean.All], frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def any(
        self, node: FExpr[ir.boolean.Any], frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def arg_max(
        self, node: agg.ArgMax, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def arg_min(
        self, node: agg.ArgMin, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def count(
        self, node: agg.Count, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def first(
        self, node: agg.First, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def last(
        self, node: agg.Last, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def len(
        self, node: agg.Len, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def max(
        self, node: agg.Max, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def mean(
        self, node: agg.Mean, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def median(
        self, node: agg.Median, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def min(
        self, node: agg.Min, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def mode_any(
        self, node: FExpr[F.ModeAny], frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def n_unique(
        self, node: agg.NUnique, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def null_count(
        self, node: FExpr[F.NullCount], frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def quantile(
        self, node: agg.Quantile, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def sum(
        self, node: agg.Sum, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def std(
        self, node: agg.Std, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def var(
        self, node: agg.Var, frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def kurtosis(
        self, node: FExpr[F.Kurtosis], frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def skew(
        self, node: FExpr[F.Skew], frame: Frame, name: str, /
    ) -> Scalar[Frame, NativeExpr_co, NativeScalar_co]: ...


class EagerColumn(
    BroadcastSeries[NativeSeriesT],
    CompliantColumn[EagerFrame, Native_co],
    Protocol[EagerFrame, Native_co, NativeSeriesT],
):
    """`[EagerDataFrameT_contra, Native_co, NativeSeriesT]`."""

    __slots__ = ()

    # TODO @dangotbanned: Weed out the source of
    #   "Avoids falling back to `__len__` when truth-testing on dispatch"
    def __bool__(self) -> Literal[True]:
        return True

    def is_in_series(
        self,
        node: FExpr[ir.boolean.IsInSeries[IncompleteCyclic]],
        frame: EagerFrame,
        name: str,
        /,
    ) -> Self: ...


class EagerExpr(
    BroadcastSeries[NativeSeriesT],
    CompliantExpr[EagerFrame, NativeExpr_co, NativeScalar_co],
    Protocol[EagerFrame, NativeExpr_co, NativeScalar_co, NativeSeriesT],
):
    """`[EagerDataFrameT_contra, NativeExpr_co, NativeScalar_co, NativeSeriesT]`."""

    __slots__ = ()

    @classmethod
    def lit_series(
        cls, node: ir.LitSeries[Any], frame: EagerFrame, name: str, /
    ) -> Self: ...
    def gather_every(
        self, node: FExpr[F.GatherEvery], frame: EagerFrame, name: str, /
    ) -> Self: ...
    def map_batches(
        self, node: ir.AnonymousExpr, frame: EagerFrame, name: str, /
    ) -> Self | EagerScalar[EagerFrame, NativeExpr_co, NativeScalar_co]: ...
    def sample_frac(
        self, node: FExpr[F.SampleFrac], frame: EagerFrame, name: str, /
    ) -> Self: ...
    def sample_n(
        self, node: FExpr[F.SampleN], frame: EagerFrame, name: str, /
    ) -> Self: ...

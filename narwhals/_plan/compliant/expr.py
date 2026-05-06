from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from narwhals._plan.compliant.broadcast import BroadcastSeries
from narwhals._plan.compliant.typing import (
    EagerDataFrameT,
    FrameT,
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
    from narwhals._plan.compliant.namespace import CompliantNamespace, EagerNamespace
    from narwhals._plan.compliant.scalar import CompliantScalar as Scalar, EagerScalar
    from narwhals._plan.expressions import (
        BinaryExpr,
        FunctionExpr as FExpr,
        aggregation as agg,
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
    from narwhals._utils import Version
    from narwhals.typing import PythonLiteral

Incomplete: TypeAlias = Any


# TODO @dangotbanned: Avoid `FrameT`?
# TODO @dangotbanned: Add a common base for `CompliantExpr`, `CompliantScalar`
# - Will resolve some incompatible overrides
# - And allow constructors to replace namespace expr methods
# TODO @dangotbanned: Namespace methods -> constructors
# - [ ] `CompliantExpr`
#   - [ ] `col`
# - [x] `CompliantScalar` (preferred, but is optional so `CompliantExpr` needs it too)
#   - [x] `lit`
#   - [x] `len_star`
# - [x] `EagerExpr`
#   - [x] `lit_series`
# TODO @dangotbanned: Namespace methods -> Expr methods
# - [ ] Binary
#   - [ ] `date_range`
#   - [ ] `int_range`
#   - [ ] `linear_space`
# - [ ] Variadic (they can technically be either)
class CompliantExpr(Protocol[FrameT, NativeExpr_co, NativeScalar_co]):
    """Everything common to `Expr` and `Scalar` literal values.

    `[FrameT, NativeExpr_co, NativeScalar_co]`.
    """

    __slots__ = ()

    version: ClassVar[Version]

    @property
    def native(self) -> NativeExpr_co: ...

    # TODO @dangotbanned: Review return once `CompliantScalar` no longer inherits from `CompliantExpr`
    @classmethod
    def len_star(
        cls, node: ir.Len, frame: FrameT, name: str, /
    ) -> Self | Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    # TODO @dangotbanned: Review return once `CompliantScalar` no longer inherits from `CompliantExpr`
    @classmethod
    def lit(
        cls, node: ir.Lit[PythonLiteral], frame: FrameT, name: str, /
    ) -> Self | Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...

    # NOTE: May need to change returning `Self` to `CompliantExpr[FrameT]`
    # Expr -> Expr
    # Scalar -> Scalar
    def abs(self, node: FExpr[F.Abs], frame: FrameT, name: str, /) -> Self: ...
    def binary_expr(self, node: BinaryExpr, frame: FrameT, name: str, /) -> Self: ...
    def cast(self, node: ir.Cast, frame: FrameT, name: str, /) -> Self: ...
    def ceil(self, node: FExpr[F.Ceil], frame: FrameT, name: str, /) -> Self: ...
    def clip(self, node: FExpr[F.Clip], frame: FrameT, name: str, /) -> Self: ...
    def clip_lower(
        self, node: FExpr[F.ClipLower], frame: FrameT, name: str, /
    ) -> Self: ...
    def clip_upper(
        self, node: FExpr[F.ClipUpper], frame: FrameT, name: str, /
    ) -> Self: ...
    def drop_nulls(
        self, node: FExpr[F.DropNulls], frame: FrameT, name: str, /
    ) -> Self: ...
    def ewm_mean(self, node: FExpr[F.EwmMean], frame: FrameT, name: str, /) -> Self: ...
    def exp(self, node: FExpr[F.Exp], frame: FrameT, name: str, /) -> Self: ...
    def fill_null(self, node: FExpr[F.FillNull], frame: FrameT, name: str, /) -> Self: ...
    def fill_null_with_strategy(
        self, node: FExpr[F.FillNullWithStrategy], frame: FrameT, name: str, /
    ) -> Self: ...
    def floor(self, node: FExpr[F.Floor], frame: FrameT, name: str, /) -> Self: ...
    def hist_bins(self, node: FExpr[F.HistBins], frame: FrameT, name: str, /) -> Self: ...
    def hist_bin_count(
        self, node: FExpr[F.HistBinCount], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_between(self, node: FExpr[IsBetween], frame: FrameT, name: str, /) -> Self: ...
    def is_duplicated(
        self, node: FExpr[ir.boolean.IsDuplicated], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_finite(self, node: FExpr[IsFinite], frame: FrameT, name: str, /) -> Self: ...
    def is_first_distinct(
        self, node: FExpr[IsFirstDistinct], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_in_expr(
        self, node: FExpr[ir.boolean.IsInExpr], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_in_seq(
        self, node: FExpr[ir.boolean.IsInSeq], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_last_distinct(
        self, node: FExpr[IsLastDistinct], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_nan(self, node: FExpr[IsNan], frame: FrameT, name: str, /) -> Self: ...
    def is_null(self, node: FExpr[IsNull], frame: FrameT, name: str, /) -> Self: ...
    def is_not_nan(self, node: FExpr[IsNotNan], frame: FrameT, name: str, /) -> Self: ...
    def is_not_null(
        self, node: FExpr[IsNotNull], frame: FrameT, name: str, /
    ) -> Self: ...
    def is_unique(
        self, node: FExpr[ir.boolean.IsUnique], frame: FrameT, name: str, /
    ) -> Self: ...
    def log(self, node: FExpr[F.Log], frame: FrameT, name: str, /) -> Self: ...
    def mode_all(self, node: FExpr[F.ModeAll], frame: FrameT, name: str, /) -> Self: ...
    def not_(self, node: FExpr[Not], frame: FrameT, name: str, /) -> Self: ...
    def over(self, node: ir.Over, frame: FrameT, name: str, /) -> Self: ...
    def pow(self, node: FExpr[F.Pow], frame: FrameT, name: str, /) -> Self: ...
    def replace_strict(
        self, node: FExpr[F.ReplaceStrict], frame: FrameT, name: str, /
    ) -> Self: ...
    def replace_strict_default(
        self, node: FExpr[F.ReplaceStrictDefault], frame: FrameT, name: str, /
    ) -> Self: ...
    def rolling_mean(
        self, node: FExpr[F.RollingMean], frame: FrameT, name: str, /
    ) -> Self: ...
    def rolling_sum(
        self, node: FExpr[F.RollingSum], frame: FrameT, name: str, /
    ) -> Self: ...
    def rolling_std(
        self, node: FExpr[F.RollingStd], frame: FrameT, name: str, /
    ) -> Self: ...
    def rolling_var(
        self, node: FExpr[F.RollingVar], frame: FrameT, name: str, /
    ) -> Self: ...
    def round(self, node: FExpr[F.Round], frame: FrameT, name: str, /) -> Self: ...
    def shift(self, node: FExpr[F.Shift], frame: FrameT, name: str, /) -> Self: ...
    def sqrt(self, node: FExpr[F.Sqrt], frame: FrameT, name: str, /) -> Self: ...
    def ternary_expr(self, node: ir.TernaryExpr, frame: FrameT, name: str, /) -> Self: ...
    def unique(self, node: FExpr[F.Unique], frame: FrameT, name: str, /) -> Self: ...

    # Expr -> Expr
    # (some are `Scalar` noops)
    def cum_count(self, node: FExpr[F.CumCount], frame: FrameT, name: str, /) -> Self: ...
    def cum_max(self, node: FExpr[F.CumMax], frame: FrameT, name: str, /) -> Self: ...
    def cum_min(self, node: FExpr[F.CumMin], frame: FrameT, name: str, /) -> Self: ...
    def cum_prod(self, node: FExpr[F.CumProd], frame: FrameT, name: str, /) -> Self: ...
    def cum_sum(self, node: FExpr[F.CumSum], frame: FrameT, name: str, /) -> Self: ...
    def diff(self, node: FExpr[F.Diff], frame: FrameT, name: str, /) -> Self: ...
    def filter(self, node: ir.Filter, frame: FrameT, name: str, /) -> Self: ...
    def rank(self, node: FExpr[F.Rank], frame: FrameT, name: str, /) -> Self: ...
    def sort(self, node: ir.Sort, frame: FrameT, name: str, /) -> Self: ...
    def sort_by(self, node: ir.SortBy, frame: FrameT, name: str, /) -> Self: ...

    # Expr -> Scalar
    # TODO @dangotbanned: Move this concept to the ExprIR layer?
    # - Every `Function` has `FunctionFlags.AGGREGATION`
    # - Everything* else is an `AggExpr`
    # - `OverOrdered` is an outlier, and it also doesn't specify `is_scalar`?
    def all(
        self, node: FExpr[ir.boolean.All], frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def any(
        self, node: FExpr[ir.boolean.Any], frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def arg_max(
        self, node: agg.ArgMax, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def arg_min(
        self, node: agg.ArgMin, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def count(
        self, node: agg.Count, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def first(
        self, node: agg.First, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def last(
        self, node: agg.Last, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def len(
        self, node: agg.Len, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def max(
        self, node: agg.Max, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def mean(
        self, node: agg.Mean, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def median(
        self, node: agg.Median, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def min(
        self, node: agg.Min, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def mode_any(
        self, node: FExpr[F.ModeAny], frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def n_unique(
        self, node: agg.NUnique, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def null_count(
        self, node: FExpr[F.NullCount], frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    # NOTE: `Scalar` is returned **only** for un-partitioned `OrderableAggExpr`
    #  - e.g. `nw.col("a").first().over(order_by="b")`
    # TODO @dangotbanned: Split (un-partitioned + ordered) into another node?
    # - The handling of the union would need repeating everything otherwise
    # - https://github.com/narwhals-dev/narwhals/blob/489bada3e9318f91c9d73744e7a6de62d2478451/narwhals/_plan/arrow/expr.py#L560-L570
    def over_ordered(
        self, node: ir.OverOrdered, frame: FrameT, name: str, /
    ) -> Self | Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def quantile(
        self, node: agg.Quantile, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def sum(
        self, node: agg.Sum, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def std(
        self, node: agg.Std, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def var(
        self, node: agg.Var, frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def kurtosis(
        self, node: FExpr[F.Kurtosis], frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...
    def skew(
        self, node: FExpr[F.Skew], frame: FrameT, name: str, /
    ) -> Scalar[FrameT, NativeExpr_co, NativeScalar_co]: ...

    def __narwhals_namespace__(
        self,
    ) -> CompliantNamespace[
        FrameT,
        CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co],
        CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co],
    ]: ...

    def __narwhals_expr_prepare__(self) -> Self:
        """Return a partially initialized instance of this class.

        The only external (narwhals-level) requirement is that we have an instance to call methods on.
        """
        tp = type(self)
        return tp.__new__(tp)

    @property
    def cat(
        self,
    ) -> ExprCatNamespace[
        FrameT, CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def dt(
        self,
    ) -> ExprDateTimeNamespace[
        FrameT, CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def list(
        self,
    ) -> ExprListNamespace[
        FrameT, CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def str(
        self,
    ) -> ExprStringNamespace[
        FrameT, CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def struct(
        self,
    ) -> ExprStructNamespace[
        FrameT, CompliantExpr[FrameT, NativeExpr_co, NativeScalar_co]
    ]: ...


class EagerExpr(
    BroadcastSeries[NativeSeriesT],
    CompliantExpr[EagerDataFrameT, NativeExpr_co, NativeScalar_co],
    Protocol[EagerDataFrameT, NativeExpr_co, NativeScalar_co, NativeSeriesT],
):
    """`[EagerDataFrameT, NativeExpr_co, NativeScalar_co, NativeSeriesT]`."""

    __slots__ = ()

    # TODO @dangotbanned: Change return to `Self` once `EagerScalar` no longer inherits from `EagerExpr`
    @classmethod
    def lit_series(
        cls, node: ir.LitSeries[Any], frame: EagerDataFrameT, name: str, /
    ) -> EagerExpr[EagerDataFrameT, NativeExpr_co, NativeScalar_co, NativeSeriesT]: ...

    def __bool__(self) -> Literal[True]:
        # NOTE: Avoids falling back to `__len__` when truth-testing on dispatch
        return True

    def gather_every(
        self, node: FExpr[F.GatherEvery], frame: EagerDataFrameT, name: str, /
    ) -> Self: ...
    def is_in_series(
        self,
        node: FExpr[ir.boolean.IsInSeries[IncompleteCyclic]],
        frame: EagerDataFrameT,
        name: str,
        /,
    ) -> Self: ...
    # NOTE: `Scalar` when using `returns_scalar=True`
    def map_batches(
        self, node: ir.AnonymousExpr, frame: EagerDataFrameT, name: str, /
    ) -> (
        Self | EagerScalar[EagerDataFrameT, NativeExpr_co, NativeScalar_co, NativeSeriesT]
    ): ...
    def sample_frac(
        self, node: FExpr[F.SampleFrac], frame: EagerDataFrameT, name: str, /
    ) -> Self: ...
    # NOTE: `n=1` can behave similar to an aggregation in `select(...)`, but requires `.first()`
    # to trigger broadcasting in `with_columns(...)`
    def sample_n(
        self, node: FExpr[F.SampleN], frame: EagerDataFrameT, name: str, /
    ) -> Self: ...
    def __narwhals_namespace__(
        self,
    ) -> EagerNamespace[
        EagerDataFrameT,
        Incomplete,
        EagerExpr[EagerDataFrameT, NativeExpr_co, NativeScalar_co, NativeSeriesT],
        EagerScalar[EagerDataFrameT, NativeExpr_co, NativeScalar_co, NativeSeriesT],
    ]: ...

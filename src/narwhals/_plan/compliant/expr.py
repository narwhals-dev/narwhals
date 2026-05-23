from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan.compliant.broadcast import BroadcastSeries
from narwhals._plan.compliant.typing import (
    DeprecatedFrameT_contra as Frame,
    NativeColumn_co,
    NativeExpr_co,
    NativeScalar_co,
    NativeSeriesT,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant import classes as cc, typing as ct
    from narwhals._plan.compliant.accessors import (
        ExprCatNamespace,
        ExprDateTimeNamespace,
        ExprListNamespace,
        ExprStringNamespace,
        ExprStructNamespace,
    )
    from narwhals._plan.compliant.scalar import EagerScalar
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
    from narwhals._plan.expressions.ranges import DateRange, IntRange, LinearSpace
    from narwhals._plan.typing import IncompleteCyclic, IncompleteVarianceLie
    from narwhals._utils import Version
    from narwhals.typing import PythonLiteral

Incomplete: TypeAlias = Any


if TYPE_CHECKING:
    from typing_extensions import TypeAliasType

    # NOTE: `TypeAliasType` allows specifying the order of `type_params`.
    # This is useful because it means we can spell the following types:
    #   `CompliantExpr   = ct.Column[Frame, NativeExpr_co,   NativeExpr_co, NativeScalar_co]`
    #   `CompliantScalar = ct.Column[Frame, NativeScalar_co, NativeExpr_co, NativeScalar_co]`
    # ... with less dancing around which position each parameters goes into:
    #   `Scalar[Frame, NativeExpr_co, NativeScalar_co]`
    Scalar = TypeAliasType(
        "Scalar",
        "ct.Column[Frame, NativeScalar_co, NativeExpr_co, NativeScalar_co]",
        type_params=(Frame, NativeExpr_co, NativeScalar_co),
    )

elif sys.version_info >= (3, 12):  # pragma: no cover
    from typing import TypeAliasType

    Scalar = TypeAliasType(
        "Scalar",
        "ct.Column[Frame, NativeScalar_co, NativeExpr_co, NativeScalar_co]",
        type_params=(Frame, NativeExpr_co, NativeScalar_co),
    )
else:  # pragma: no cover
    Scalar: TypeAlias = (
        "ct.Column[Frame, NativeScalar_co, NativeExpr_co, NativeScalar_co]"
    )


class CompliantColumn(Protocol[Frame, NativeColumn_co, NativeExpr_co, NativeScalar_co]):
    """Everything common to `Expr` and `Scalar` literal values.

    `[FrameT_contra, NativeColumn_co, NativeExpr_co, NativeScalar_co]`.
    """

    __slots__ = ()
    version: ClassVar[Version]

    @property
    def native(self) -> NativeColumn_co: ...

    # Constructors (Scalar)  # noqa: ERA001
    @classmethod
    def len_star(
        cls, node: ir.Len, frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    @classmethod
    def lit(
        cls, node: ir.Lit[PythonLiteral], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...

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
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
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
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def any_horizontal(
        self, node: HExpr[ir.boolean.AnyHorizontal], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def coalesce(
        self, node: HExpr[F.Coalesce], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def concat_str(
        self, node: HExpr[strings.ConcatStr], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def max_horizontal(
        self, node: HExpr[F.MaxHorizontal], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def min_horizontal(
        self, node: HExpr[F.MinHorizontal], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...
    def sum_horizontal(
        self, node: HExpr[F.SumHorizontal], frame: Frame, name: str, /
    ) -> Self | ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]: ...

    # Range (technically, only valid to call on `*Scalar` and then produces `*Expr`)
    def int_range(
        self, node: FExpr[IntRange], frame: Frame, name: str, /
    ) -> CompliantExpr[Frame, Incomplete, Incomplete]: ...
    def date_range(
        self, node: FExpr[DateRange], frame: Frame, name: str, /
    ) -> CompliantExpr[Frame, Incomplete, Incomplete]: ...
    def linear_space(
        self, node: FExpr[LinearSpace], frame: Frame, name: str, /
    ) -> CompliantExpr[Frame, Incomplete, Incomplete]: ...

    @property
    def __narwhals_classes__(
        self,
    ) -> cc.CompliantClasses[IncompleteCyclic, IncompleteCyclic]: ...

    def dispatch(
        self, node: ir.ExprIR, frame: IncompleteVarianceLie, name: str, /
    ) -> ct.Column[Any, Any, NativeExpr_co, NativeScalar_co]:
        """Evaluate an expression.

        Arguments:
            node: The expression to dispatch.
            frame: A`Compliant*Frame` that shares the same backend as `self`.

                Many methods will not need to use this *directly*, and simply pass the `frame`
                they *received* down recursively, for those that do need it.

                If a method needs to mutate `frame` (e.g. adding temporary columns), the mutated
                version **must not** be passed down.

            name: Output column name, which will typically have originated from `NamedIR.name`.

                When a method handles a unary expression, `name` is just passed down.

                Otherwise, pass it down for the left-most expression and use `""` for all others.

        ## Tip
        Implementing this is easy, but providing accurate typing *inside* the protocol is not (*cries, recursively*).

        It should look something like this:

            def dispatch(self, node: ExprIR, frame: Frame, name: str, /) -> Expr | Scalar:
                return node.__expr_ir_dispatch__(node, self, frame, name)
        """
        ...

    @property
    def cat(
        self,
    ) -> ExprCatNamespace[
        Frame, ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def dt(
        self,
    ) -> ExprDateTimeNamespace[
        Frame, ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def list(
        self,
    ) -> ExprListNamespace[
        Frame, ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]
    ]: ...
    @property
    def str(
        self,
    ) -> ExprStringNamespace[
        Frame, ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]
    ]: ...

    # TODO @dangotbanned: Migrate everything else from `DeprecatedFrameT_contra` -> `FrameT_contra`
    # `ExprStructNamespace` is done
    @property
    def struct(
        self,
    ) -> ExprStructNamespace[  # type: ignore[type-var]
        Frame, ct.Column[Frame, Incomplete, NativeExpr_co, NativeScalar_co]  # pyright: ignore[reportInvalidTypeArguments]
    ]: ...


class CompliantExpr(
    CompliantColumn[Frame, NativeExpr_co, NativeExpr_co, NativeScalar_co],
    Protocol[Frame, NativeExpr_co, NativeScalar_co],
):
    """`[FrameT_contra, NativeExpr_co, NativeScalar_co]`."""

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
    CompliantColumn[Frame, NativeColumn_co, NativeExpr_co, NativeScalar_co],
    Protocol[Frame, NativeColumn_co, NativeExpr_co, NativeScalar_co, NativeSeriesT],
):
    """`[FrameT_contra, NativeColumn_co, NativeExpr_co, NativeScalar_co, NativeSeriesT]`."""

    __slots__ = ()

    def is_in_series(
        self,
        node: FExpr[ir.boolean.IsInSeries[IncompleteCyclic]],
        frame: Frame,
        name: str,
        /,
    ) -> Self: ...


class EagerExpr(
    BroadcastSeries[NativeSeriesT],
    CompliantExpr[Frame, NativeExpr_co, NativeScalar_co],
    Protocol[Frame, NativeExpr_co, NativeScalar_co, NativeSeriesT],
):
    """`[FrameT_contra, NativeExpr_co, NativeScalar_co, NativeSeriesT]`."""

    __slots__ = ()

    @classmethod
    def lit_series(cls, node: ir.LitSeries[Any], frame: Frame, name: str, /) -> Self: ...
    def gather_every(
        self, node: FExpr[F.GatherEvery], frame: Frame, name: str, /
    ) -> Self: ...
    def map_batches(
        self, node: ir.AnonymousExpr, frame: Frame, name: str, /
    ) -> Self | EagerScalar[Frame, NativeExpr_co, NativeScalar_co]: ...
    def sample_frac(
        self, node: FExpr[F.SampleFrac], frame: Frame, name: str, /
    ) -> Self: ...
    def sample_n(self, node: FExpr[F.SampleN], frame: Frame, name: str, /) -> Self: ...

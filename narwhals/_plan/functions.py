"""General functions that aren't namespaced."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Function
from narwhals._plan.exceptions import hist_bins_monotonic_error
from narwhals._plan.options import FunctionFlags, FunctionOptions

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.expr import AnonymousExpr, RollingExpr
    from narwhals._plan.options import EWMOptions, RankOptions, RollingOptionsFixedWindow
    from narwhals._plan.typing import Seq, Udf
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy


class Abs(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "abs"


class Hist(Function):
    """Only supported for `Series` so far."""

    __slots__ = ("include_breakpoint",)
    include_breakpoint: bool

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "hist"


class HistBins(Hist):
    __slots__ = ("bins", *Hist.__slots__)
    bins: Seq[float]

    def __init__(self, *, bins: Seq[float], include_breakpoint: bool = True) -> None:
        for i in range(1, len(bins)):
            if bins[i - 1] >= bins[i]:
                raise hist_bins_monotonic_error(bins)
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "include_breakpoint", include_breakpoint)


class HistBinCount(Hist):
    __slots__ = ("bin_count", *Hist.__slots__)
    bin_count: int
    """Polars (v1.20) sets `bin_count=10` if neither `bins` or `bin_count` are provided."""

    def __init__(self, *, bin_count: int = 10, include_breakpoint: bool = True) -> None:
        object.__setattr__(self, "bin_count", bin_count)
        object.__setattr__(self, "include_breakpoint", include_breakpoint)


class NullCount(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()

    def __repr__(self) -> str:
        return "null_count"


class Log(Function):
    __slots__ = ("base",)
    base: float

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "log"


class Exp(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "exp"


class Pow(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "pow"


class Sqrt(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "sqrt"


class Kurtosis(Function):
    __slots__ = ("bias", "fisher")
    fisher: bool
    bias: bool

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()

    def __repr__(self) -> str:
        return "kurtosis"


class FillNull(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "fill_null"


class FillNullWithStrategy(Function):
    """We don't support this variant in a lot of backends, so worth keeping it split out.

    https://github.com/narwhals-dev/narwhals/pull/2555
    """

    __slots__ = ("limit", "strategy")
    strategy: FillNullStrategy
    limit: int | None

    @property
    def function_options(self) -> FunctionOptions:
        # NOTE: We don't support these strategies yet
        # but might be good to encode this difference now
        return (
            FunctionOptions.elementwise()
            if self.strategy in {"one", "zero"}
            else FunctionOptions.groupwise()
        )

    def __repr__(self) -> str:
        return "fill_null_with_strategy"


class Shift(Function):
    __slots__ = ("n",)
    n: int

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()

    def __repr__(self) -> str:
        return "shift"


class DropNulls(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.row_separable()

    def __repr__(self) -> str:
        return "drop_nulls"


class Mode(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "mode"


class Skew(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()

    def __repr__(self) -> str:
        return "skew"


class Rank(Function):
    __slots__ = ("options",)
    options: RankOptions

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "rank"


class Clip(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "clip"


class CumAgg(Function):
    __slots__ = ("reverse",)
    reverse: bool

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()

    def __repr__(self) -> str:
        tp = type(self)
        if tp is CumAgg:
            return tp.__name__
        m: dict[type[CumAgg], str] = {
            CumCount: "count",
            CumMin: "min",
            CumMax: "max",
            CumProd: "prod",
            CumSum: "sum",
        }
        return f"cum_{m[tp]}"


class RollingWindow(Function):
    __slots__ = ("options",)
    options: RollingOptionsFixedWindow

    @property
    def function_options(self) -> FunctionOptions:
        """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/mod.rs#L1276."""
        return FunctionOptions.length_preserving()

    def __repr__(self) -> str:
        tp = type(self)
        if tp is RollingWindow:
            return tp.__name__
        m: dict[type[RollingWindow], str] = {
            RollingSum: "sum",
            RollingMean: "mean",
            RollingVar: "var",
            RollingStd: "std",
        }
        return f"rolling_{m[tp]}"

    def to_function_expr(self, *inputs: ExprIR) -> RollingExpr[Self]:
        from narwhals._plan.expr import RollingExpr

        options = self.function_options
        return RollingExpr(input=inputs, function=self, options=options)


class CumCount(CumAgg): ...


class CumMin(CumAgg): ...


class CumMax(CumAgg): ...


class CumProd(CumAgg): ...


class CumSum(CumAgg): ...


class RollingSum(RollingWindow): ...


class RollingMean(RollingWindow): ...


class RollingVar(RollingWindow): ...


class RollingStd(RollingWindow): ...


class Diff(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()

    def __repr__(self) -> str:
        return "diff"


class Unique(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "unique"


class Round(Function):
    __slots__ = ("decimals",)
    decimals: int

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "round"


class SumHorizontal(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )

    def __repr__(self) -> str:
        return "sum_horizontal"


class MinHorizontal(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )

    def __repr__(self) -> str:
        return "min_horizontal"


class MaxHorizontal(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )

    def __repr__(self) -> str:
        return "max_horizontal"


class MeanHorizontal(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )

    def __repr__(self) -> str:
        return "mean_horizontal"


class EwmMean(Function):
    __slots__ = ("options",)
    options: EWMOptions

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()

    def __repr__(self) -> str:
        return "ewm_mean"


class ReplaceStrict(Function):
    __slots__ = ("new", "old", "return_dtype")
    old: Seq[Any]
    new: Seq[Any]
    return_dtype: DType | None

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()

    def __repr__(self) -> str:
        return "replace_strict"


class GatherEvery(Function):
    __slots__ = ("n", "offset")
    n: int
    offset: int

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()

    def __repr__(self) -> str:
        return "gather_every"


class MapBatches(Function):
    __slots__ = ("function", "is_elementwise", "return_dtype", "returns_scalar")
    function: Udf
    return_dtype: DType | None
    is_elementwise: bool
    returns_scalar: bool

    @property
    def function_options(self) -> FunctionOptions:
        """https://github.com/narwhals-dev/narwhals/issues/2522."""
        options = super().function_options
        if self.is_elementwise:
            options = options.with_elementwise()
        if self.returns_scalar:
            options = options.with_flags(FunctionFlags.RETURNS_SCALAR)
        return options

    def __repr__(self) -> str:
        return "map_batches"

    def to_function_expr(self, *inputs: ExprIR) -> AnonymousExpr:
        from narwhals._plan.expr import AnonymousExpr

        options = self.function_options
        return AnonymousExpr(input=inputs, function=self, options=options)

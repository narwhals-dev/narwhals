"""General functions that aren't namespaced."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan.exceptions import hist_bins_monotonic_error
from narwhals._plan.options import FunctionFlags, FunctionOptions

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from _typeshed import ConvertibleToInt
    from typing_extensions import Self

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.expressions.expr import AnonymousExpr, FunctionExpr, RollingExpr
    from narwhals._plan.options import EWMOptions, RankOptions, RollingOptionsFixedWindow
    from narwhals._plan.typing import Seq, Udf
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy


class CumAgg(Function, options=FunctionOptions.length_preserving):
    __slots__ = ("reverse",)
    reverse: bool


class RollingWindow(Function, options=FunctionOptions.length_preserving):
    __slots__ = ("options",)
    options: RollingOptionsFixedWindow

    def to_function_expr(self, *inputs: ExprIR) -> RollingExpr[Self]:
        from narwhals._plan.expressions.expr import RollingExpr

        options = self.function_options
        return RollingExpr(input=inputs, function=self, options=options)


# fmt: off
class Abs(Function, options=FunctionOptions.elementwise): ...
class NullCount(Function, options=FunctionOptions.aggregation): ...
class Exp(Function, options=FunctionOptions.elementwise): ...
class Sqrt(Function, options=FunctionOptions.elementwise): ...
class DropNulls(Function, options=FunctionOptions.row_separable): ...
class Mode(Function): ...
class Skew(Function, options=FunctionOptions.aggregation): ...
class Clip(Function, options=FunctionOptions.elementwise): ...
class CumCount(CumAgg): ...
class CumMin(CumAgg): ...
class CumMax(CumAgg): ...
class CumProd(CumAgg): ...
class CumSum(CumAgg): ...
class RollingSum(RollingWindow): ...
class RollingMean(RollingWindow): ...
class RollingVar(RollingWindow): ...
class RollingStd(RollingWindow): ...
class Diff(Function, options=FunctionOptions.length_preserving): ...
class Unique(Function): ...
class SumHorizontal(HorizontalFunction): ...
class MinHorizontal(HorizontalFunction): ...
class MaxHorizontal(HorizontalFunction): ...
class MeanHorizontal(HorizontalFunction): ...
# fmt: on
class Hist(Function):
    """Only supported for `Series` so far."""

    __slots__ = ("include_breakpoint",)
    include_breakpoint: bool

    def __repr__(self) -> str:
        return "hist"

    # NOTE: These constructors provide validation + defaults, and avoid
    # repeating on every `__init__` afterwards
    # They're also more widely defined to what will work at runtime
    @staticmethod
    def from_bins(
        bins: Iterable[float], /, *, include_breakpoint: bool = True
    ) -> HistBins:
        bins = tuple(bins)
        for i in range(1, len(bins)):
            if bins[i - 1] >= bins[i]:
                raise hist_bins_monotonic_error(bins)
        return HistBins(bins=bins, include_breakpoint=include_breakpoint)

    @staticmethod
    def from_bin_count(
        count: ConvertibleToInt = 10, /, *, include_breakpoint: bool = True
    ) -> HistBinCount:
        return HistBinCount(bin_count=int(count), include_breakpoint=include_breakpoint)


class HistBins(Hist):
    __slots__ = ("bins",)
    bins: Seq[float]


class HistBinCount(Hist):
    __slots__ = ("bin_count",)
    bin_count: int


class Log(Function, options=FunctionOptions.elementwise):
    __slots__ = ("base",)
    base: float


class Pow(Function, options=FunctionOptions.elementwise):
    """N-ary (base, exponent)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        base, exponent = node.input
        return base, exponent


class Kurtosis(Function, options=FunctionOptions.aggregation):
    __slots__ = ("bias", "fisher")
    fisher: bool
    bias: bool


class FillNull(Function, options=FunctionOptions.elementwise):
    """N-ary (expr, value)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, value = node.input
        return expr, value


class FillNullWithStrategy(Function):
    __slots__ = ("limit", "strategy")
    strategy: FillNullStrategy
    limit: int | None


class Shift(Function, options=FunctionOptions.length_preserving):
    __slots__ = ("n",)
    n: int


class Rank(Function):
    __slots__ = ("options",)
    options: RankOptions


class Round(Function, options=FunctionOptions.elementwise):
    __slots__ = ("decimals",)
    decimals: int


class EwmMean(Function, options=FunctionOptions.length_preserving):
    __slots__ = ("options",)
    options: EWMOptions


class ReplaceStrict(Function, options=FunctionOptions.elementwise):
    __slots__ = ("new", "old", "return_dtype")
    old: Seq[Any]
    new: Seq[Any]
    return_dtype: DType | None


class GatherEvery(Function):
    __slots__ = ("n", "offset")
    n: int
    offset: int


class MapBatches(Function):
    __slots__ = ("function", "is_elementwise", "return_dtype", "returns_scalar")
    function: Udf
    return_dtype: DType | None
    is_elementwise: bool
    returns_scalar: bool

    @property
    def function_options(self) -> FunctionOptions:
        options = super().function_options
        if self.is_elementwise:
            options = options.with_elementwise()
        if self.returns_scalar:
            options = options.with_flags(FunctionFlags.RETURNS_SCALAR)
        return options

    def to_function_expr(self, *inputs: ExprIR) -> AnonymousExpr:
        from narwhals._plan.expressions.expr import AnonymousExpr

        options = self.function_options
        return AnonymousExpr(input=inputs, function=self, options=options)

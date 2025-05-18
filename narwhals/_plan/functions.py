from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR
from narwhals._plan.common import Function
from narwhals._plan.options import FunctionFlags
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.common import Seq
    from narwhals._plan.common import Udf
    from narwhals._plan.options import EWMOptions
    from narwhals._plan.options import RankOptions
    from narwhals._plan.options import RollingOptionsFixedWindow
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy


# TODO @dangotbanned: repr


class Abs(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class Hist(Function):
    """Only supported for `Series` so far."""

    __slots__ = ("include_breakpoint",)

    include_breakpoint: bool

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()


class HistBins(Hist):
    """Subclasses for each variant."""

    __slots__ = (*Hist.__slots__, "bins")

    bins: Seq[float]


class HistBinCount(Hist):
    __slots__ = (*Hist.__slots__, "bin_count")

    bin_count: int


class NullCount(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()


class Pow(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class FillNull(Function):
    __slots__ = ("value",)

    value: ExprIR

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


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


class Shift(Function):
    __slots__ = ("n",)

    n: int
    """https://github.com/narwhals-dev/narwhals/pull/2555"""

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class DropNulls(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.row_separable()


class Mode(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()


class Skew(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.aggregation()


class Rank(Function):
    __slots__ = ("options",)

    options: RankOptions

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()


class Clip(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class CumAgg(Function):
    __slots__ = ("reverse",)

    reverse: bool
    """https://github.com/narwhals-dev/narwhals/pull/2555"""

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class RollingWindow(Function):
    __slots__ = ("options",)

    options: RollingOptionsFixedWindow

    @property
    def function_options(self) -> FunctionOptions:
        """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/function_expr/mod.rs#L1276."""
        return FunctionOptions.length_preserving()


class CumCount(CumAgg): ...


class CumMin(CumAgg): ...


class CumMax(CumAgg): ...


class CumProd(CumAgg): ...


class RollingSum(RollingWindow): ...


class RollingMean(RollingWindow): ...


class RollingVar(RollingWindow): ...


class RollingStd(RollingWindow): ...


class Diff(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.length_preserving()


class Unique(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()


class Round(Function):
    __slots__ = ("decimals",)

    decimals: int

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


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


class ReplaceStrict(Function):
    __slots__ = ("return_dtype",)

    return_dtype: DType | type[DType] | None

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.elementwise()


class GatherEvery(Function):
    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.groupwise()


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

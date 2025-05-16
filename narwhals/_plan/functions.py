"""TODO: Attributes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR
from narwhals._plan.common import Function
from narwhals._plan.options import FunctionFlags
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from typing import Any
    from typing import Sequence

    from narwhals._plan.options import EWMOptions
    from narwhals._plan.options import RankOptions
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy


class Abs(Function): ...


class Hist(Function):
    """Only supported for `Series` so far."""

    __slots__ = ("include_breakpoint",)

    include_breakpoint: bool


class HistBins(Hist):
    """Subclasses for each variant."""

    __slots__ = (*Hist.__slots__, "bins")

    bins: Sequence[float]


class HistBinCount(Hist):
    __slots__ = (*Hist.__slots__, "bin_count")

    bin_count: int


class NullCount(Function): ...


class Pow(Function): ...


class FillNull(Function):
    __slots__ = ("value",)

    value: ExprIR


class FillNullWithStrategy(Function):
    """We don't support this variant in a lot of backends, so worth keeping it split out."""

    __slots__ = ("limit", "strategy")

    strategy: FillNullStrategy
    limit: int | None


class Shift(Function): ...


class DropNulls(Function): ...


class Mode(Function): ...


class Skew(Function): ...


class Rank(Function):
    __slots__ = ("options",)

    options: RankOptions


class Clip(Function): ...


class CumAgg(Function):
    __slots__ = ("reverse",)

    reverse: bool


class CumCount(CumAgg): ...


class CumMin(CumAgg): ...


class CumMax(CumAgg): ...


class CumProd(CumAgg): ...


class Diff(Function): ...


class Unique(Function): ...


class Round(Function):
    __slots__ = ("decimals",)

    decimals: int


class SumHorizontal(Function): ...


class MinHorizontal(Function): ...


class MaxHorizontal(Function): ...


class MeanHorizontal(Function): ...


class EwmMean(Function):
    __slots__ = ("options",)

    options: EWMOptions


class ReplaceStrict(Function):
    __slots__ = ("return_dtype",)

    return_dtype: DType | type[DType] | None


class GatherEvery(Function): ...


class MapBatches(Function):
    __slots__ = ("function", "is_elementwise", "return_dtype", "returns_scalar")

    function: Any
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

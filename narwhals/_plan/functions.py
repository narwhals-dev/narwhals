"""TODO: Attributes."""

from __future__ import annotations

from narwhals._plan.common import Function


class Abs(Function): ...


class Hist(Function):
    """Only supported for `Series` so far."""


class NullCount(Function): ...


class Pow(Function): ...


class FillNull(Function): ...


class FillNullWithStrategy(Function):
    """We don't support this variant in a lot of backends, so worth keeping it split out."""


class Shift(Function): ...


class DropNulls(Function): ...


class Mode(Function): ...


class Skew(Function): ...


class Rank(Function): ...


class Clip(Function): ...


class CumCount(Function): ...


class CumMin(Function): ...


class CumMax(Function): ...


class CumProd(Function): ...


class Diff(Function): ...


class Unique(Function): ...


class Round(Function): ...


class SumHorizontal(Function): ...


class MinHorizontal(Function): ...


class MaxHorizontal(Function): ...


class MeanHorizontal(Function): ...


class EwmMean(Function): ...


class ReplaceStrict(Function): ...


class GatherEvery(Function): ...


class MapBatches(Function): ...

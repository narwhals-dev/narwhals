from __future__ import annotations

from narwhals._plan.common import FunctionExpr


class Abs(FunctionExpr): ...


class Hist(FunctionExpr):
    """Only supported for `Series` so far."""


class NullCount(FunctionExpr): ...


class Pow(FunctionExpr): ...


class FillNull(FunctionExpr): ...


class FillNullWithStrategy(FunctionExpr):
    """We don't support this variant in a lot of backends, so worth keeping it split out."""


class Shift(FunctionExpr): ...


class DropNulls(FunctionExpr): ...


class Mode(FunctionExpr): ...


class Skew(FunctionExpr): ...


class Rank(FunctionExpr): ...


class Clip(FunctionExpr): ...


class CumCount(FunctionExpr): ...


class CumMin(FunctionExpr): ...


class CumMax(FunctionExpr): ...


class CumProd(FunctionExpr): ...


class Diff(FunctionExpr): ...


class Unique(FunctionExpr): ...


class Round(FunctionExpr): ...


class SumHorizontal(FunctionExpr): ...


class MinHorizontal(FunctionExpr): ...


class MaxHorizontal(FunctionExpr): ...


class MeanHorizontal(FunctionExpr): ...


class EwmMean(FunctionExpr): ...


class ReplaceStrict(FunctionExpr): ...


class GatherEvery(FunctionExpr): ...


class MapBatches(FunctionExpr): ...

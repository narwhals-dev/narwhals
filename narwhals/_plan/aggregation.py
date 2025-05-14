from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR

if TYPE_CHECKING:
    from narwhals.typing import RollingInterpolationMethod


class Agg(ExprIR):
    expr: ExprIR


class Count(Agg): ...


class First(Agg):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Last(Agg):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Max(Agg): ...


class Mean(Agg): ...


class Median(Agg): ...


class Min(Agg): ...


class NUnique(Agg): ...


class Quantile(Agg):
    quantile: ExprIR
    interpolation: RollingInterpolationMethod


class Std(Agg):
    ddof: int


class Sum(Agg): ...


class Var(Agg):
    ddof: int


class OrderableAgg(Agg): ...


class ArgMin(OrderableAgg): ...


class ArgMax(OrderableAgg): ...

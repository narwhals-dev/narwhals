from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR

if TYPE_CHECKING:
    from narwhals.typing import RollingInterpolationMethod


class Agg(ExprIR):
    __slots__ = ("expr",)

    expr: ExprIR

    @property
    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        tp = type(self)
        if tp in {Agg, OrderableAgg}:
            return tp.__name__
        m = {ArgMin: "arg_min", ArgMax: "arg_max", NUnique: "n_unique"}
        name = m.get(tp, tp.__name__.lower())
        return f"{self.expr!r}.{name}()"


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
    __slots__ = (*Agg.__slots__, "interpolation", "quantile")

    quantile: ExprIR
    interpolation: RollingInterpolationMethod


class Std(Agg):
    __slots__ = (*Agg.__slots__, "ddof")

    ddof: int


class Sum(Agg): ...


class Var(Agg):
    __slots__ = (*Agg.__slots__, "ddof")

    ddof: int


class OrderableAgg(Agg): ...


class ArgMin(OrderableAgg): ...


class ArgMax(OrderableAgg): ...

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.common import ExprIR
from narwhals._plan.exceptions import agg_scalar_error

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan.typing import MapIR
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

    def iter_left(self) -> Iterator[ExprIR]:
        yield from self.expr.iter_left()
        yield self

    def iter_right(self) -> Iterator[ExprIR]:
        yield self
        yield from self.expr.iter_right()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def map_ir(self, function: MapIR, /) -> ExprIR:
        return function(self.with_expr(self.expr.map_ir(function)))

    def with_expr(self, expr: ExprIR, /) -> Self:
        if expr == self.expr:
            return self
        it = ((k, v) for k, v in self.__immutable_items__ if k != "expr")
        return type(self)(expr=expr, **dict(it))

    def __init__(self, *, expr: ExprIR, **kwds: Any) -> None:
        if expr.is_scalar:
            raise agg_scalar_error(self, expr)
        super().__init__(expr=expr, **kwds)  # pyright: ignore[reportCallIssue]


class Count(Agg): ...


class Max(Agg): ...


class Mean(Agg): ...


class Median(Agg): ...


class Min(Agg): ...


class NUnique(Agg): ...


class Quantile(Agg):
    __slots__ = (*Agg.__slots__, "interpolation", "quantile")

    quantile: float
    interpolation: RollingInterpolationMethod


class Std(Agg):
    __slots__ = (*Agg.__slots__, "ddof")
    ddof: int


class Sum(Agg): ...


class Var(Agg):
    __slots__ = (*Agg.__slots__, "ddof")
    ddof: int


class OrderableAgg(Agg): ...


class First(OrderableAgg):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class Last(OrderableAgg):
    """https://github.com/narwhals-dev/narwhals/issues/2526."""


class ArgMin(OrderableAgg): ...


class ArgMax(OrderableAgg): ...

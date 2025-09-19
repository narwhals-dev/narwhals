from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan._expr_ir import ExprIR
from narwhals._plan.common import pascal_to_snake_case
from narwhals._plan.exceptions import agg_scalar_error

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals.typing import RollingInterpolationMethod


class AggExpr(ExprIR, child=("expr",)):
    __slots__ = ("expr",)
    expr: ExprIR

    @property
    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.expr!r}.{pascal_to_snake_case(type(self).__name__)}()"

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    def __init__(self, *, expr: ExprIR, **kwds: Any) -> None:
        if expr.is_scalar:
            raise agg_scalar_error(self, expr)
        super().__init__(expr=expr, **kwds)  # pyright: ignore[reportCallIssue]


# fmt: off
class Count(AggExpr):
    """Non-null count."""
class Len(AggExpr):
    """Null-inclusive count."""
class Max(AggExpr): ...
class Mean(AggExpr): ...
class Median(AggExpr): ...
class Min(AggExpr): ...
class NUnique(AggExpr): ...
class Sum(AggExpr): ...
class OrderableAggExpr(AggExpr): ...
class First(OrderableAggExpr): ...
class Last(OrderableAggExpr): ...
class ArgMin(OrderableAggExpr): ...
class ArgMax(OrderableAggExpr): ...
# fmt: on
class Quantile(AggExpr):
    __slots__ = (*AggExpr.__slots__, "interpolation", "quantile")
    quantile: float
    interpolation: RollingInterpolationMethod


class Std(AggExpr):
    __slots__ = (*AggExpr.__slots__, "ddof")
    ddof: int


class Var(AggExpr):
    __slots__ = (*AggExpr.__slots__, "ddof")
    ddof: int

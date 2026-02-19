from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR
from narwhals._plan.exceptions import agg_scalar_error

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals.typing import RollingInterpolationMethod

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
map_first = ResolveDType.expr_ir.map_first
same_dtype = ResolveDType.expr_ir.same_dtype


class AggExpr(ExprIR, child=("expr",)):
    __slots__ = ("expr",)
    expr: ExprIR

    @property
    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.expr!r}.{self.__expr_ir_dispatch__.name}()"

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield from self.expr.iter_output_name()

    # NOTE: Interacting badly with `pyright` synthesizing the `__replace__` signature
    if not TYPE_CHECKING:

        def __init__(self, *, expr: ExprIR, **kwds: Any) -> None:
            if expr.is_scalar:
                raise agg_scalar_error(self, expr)
            super().__init__(expr=expr, **kwds)  # pyright: ignore[reportCallIssue]
    else:  # pragma: no cover
        ...


# fmt: off
class _MomentAggExpr(AggExpr, dtype=map_first(dtm.moment_dtype)): ...
class Count(AggExpr, dtype=dtm.IDX_DTYPE):
    """Non-null count."""
class Len(AggExpr, dtype=dtm.IDX_DTYPE):
    """Null-inclusive count."""
class Max(AggExpr, dtype=same_dtype()): ...
class Mean(_MomentAggExpr): ...
class Median(_MomentAggExpr): ...
class Min(AggExpr, dtype=same_dtype()): ...
class NUnique(AggExpr, dtype=dtm.IDX_DTYPE): ...
class Sum(AggExpr, dtype=map_first(dtm.sum_dtype)): ...
class OrderableAggExpr(AggExpr): ...
class First(OrderableAggExpr, dtype=same_dtype()): ...
class Last(OrderableAggExpr, dtype=same_dtype()): ...
class ArgMin(OrderableAggExpr, dtype=dtm.IDX_DTYPE): ...
class ArgMax(OrderableAggExpr, dtype=dtm.IDX_DTYPE): ...
class Quantile(_MomentAggExpr):
    __slots__ = ("interpolation", "quantile")
    quantile: float
    interpolation: RollingInterpolationMethod
class Std(_MomentAggExpr):
    __slots__ = ("ddof",)
    ddof: int
class Var(AggExpr, dtype=map_first(dtm.var_dtype)):
    __slots__ = ("ddof",)
    ddof: int
# fmt: on


def min(name: str, /) -> Min:
    from narwhals._plan.expressions import col

    return Min(expr=col(name))


def max(name: str, /) -> Max:
    from narwhals._plan.expressions import col

    return Max(expr=col(name))

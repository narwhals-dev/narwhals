from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._function import Function
from narwhals._plan.options import FEOptions, FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, RangeExpr
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval


class RangeFunction(Function, config=FEOptions.namespaced()):
    def to_function_expr(self, *inputs: ExprIR) -> RangeExpr[Self]:
        from narwhals._plan.expressions.expr import RangeExpr

        return RangeExpr(input=inputs, function=self, options=self.function_options)

    def unwrap_input(self, node: RangeExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        start, end = node.input
        return start, end


class IntRange(RangeFunction, options=FunctionOptions.row_separable):
    """N-ary (start, end)."""

    __slots__ = ("step", "dtype")  # noqa: RUF023
    step: int
    dtype: IntegerType


class DateRange(RangeFunction, options=FunctionOptions.row_separable):
    """N-ary (start, end)."""

    __slots__ = ("interval", "closed")  # noqa: RUF023
    interval: int
    closed: ClosedInterval

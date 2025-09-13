from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._function import Function
from narwhals._plan.options import FEOptions, FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.expressions.expr import RangeExpr
    from narwhals.dtypes import IntegerType


class RangeFunction(Function, config=FEOptions.namespaced()):
    def to_function_expr(self, *inputs: ExprIR) -> RangeExpr[Self]:
        from narwhals._plan.expressions.expr import RangeExpr

        return RangeExpr(input=inputs, function=self, options=self.function_options)


class IntRange(RangeFunction, options=FunctionOptions.row_separable):
    """N-ary (start, end)."""

    __slots__ = ("step", "dtype")  # noqa: RUF023
    step: int
    dtype: IntegerType

    def unwrap_input(self, node: RangeExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        start, end = node.input
        return start, end

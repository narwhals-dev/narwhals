from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._function import Function
from narwhals._plan.options import FEOptions, FunctionOptions
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, RangeExpr
    from narwhals._plan.expressions.expr import FunctionExpr
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType, IntegerType
    from narwhals.typing import ClosedInterval

dtypes = Version.MAIN.dtypes
_FLOAT_32 = dtypes.Float32
F64 = dtypes.Float64()


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

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return self.dtype


class DateRange(RangeFunction, options=FunctionOptions.row_separable):
    """N-ary (start, end)."""

    __slots__ = ("interval", "closed")  # noqa: RUF023
    interval: int
    closed: ClosedInterval

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.DATE_DTYPE


class LinearSpace(RangeFunction, options=FunctionOptions.row_separable):
    """N-ary (start, end)."""

    __slots__ = ("num_samples", "closed")  # noqa: RUF023
    num_samples: int
    closed: ClosedInterval

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        start = node.input[0]._resolve_dtype(schema)
        end = node.input[1]._resolve_dtype(schema)
        if isinstance(start, _FLOAT_32) and isinstance(end, _FLOAT_32):
            return start
        return F64

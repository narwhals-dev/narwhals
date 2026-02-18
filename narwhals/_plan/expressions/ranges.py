from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._function import Function
from narwhals._plan.options import FEOptions, FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, RangeExpr
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval


# TODO @dangotbanned: Review upstream fix https://github.com/pola-rs/polars/pull/26549
class RangeFunction(
    Function, options=FunctionOptions.row_separable, config=FEOptions.namespaced()
):
    def to_function_expr(self, *inputs: ExprIR) -> RangeExpr[Self]:
        from narwhals._plan.expressions.expr import RangeExpr

        return RangeExpr(input=inputs, function=self, options=self.function_options)

    def unwrap_input(self, node: RangeExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        start, end = node.input
        return start, end


class IntRange(RangeFunction, dtype=ResolveDType.get_dtype()):
    """N-ary (start, end)."""

    __slots__ = ("step", "dtype")  # noqa: RUF023
    step: int
    dtype: IntegerType


class DateRange(RangeFunction, dtype=dtm.DATE):
    """N-ary (start, end)."""

    __slots__ = ("interval", "closed")  # noqa: RUF023
    interval: int
    closed: ClosedInterval


class LinearSpace(RangeFunction, dtype=ResolveDType.function_map_all(dtm.floats_dtype)):
    """N-ary (start, end)."""

    __slots__ = ("num_samples", "closed")  # noqa: RUF023
    num_samples: int
    closed: ClosedInterval

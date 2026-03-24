from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._function import Function
from narwhals._plan.exceptions import range_expr_non_scalar_error
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expressions import ExprIR, RangeExpr
    from narwhals._plan.typing import Seq
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype
map_all = ResolveDType.function.map_all
namespaced = DispatcherOptions.namespaced


class RangeFunction(Function, dispatch=namespaced()):
    def _validate_input(self, input: Seq[ExprIR], /) -> Seq[ExprIR]:  # noqa: A002
        if len(input) < 2:
            msg = f"Expected at least 2 inputs for `{self!r}()`, but got `{len(input)}`.\n`{input}`"
            raise InvalidOperationError(msg)
        if not all(e.is_scalar() for e in input):
            raise range_expr_non_scalar_error(input, self)
        return input

    def to_function_expr(self, *inputs: ExprIR) -> RangeExpr[Self]:
        from narwhals._plan.expressions.expr import RangeExpr

        return RangeExpr(input=self._validate_input(inputs), function=self)

    def unwrap_input(self, node: RangeExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        start, end = node.input
        return start, end


class IntRange(RangeFunction, dtype=get_dtype()):
    """N-ary (start, end)."""

    __slots__ = ("step", "dtype")  # noqa: RUF023
    step: int
    dtype: IntegerType


class DateRange(RangeFunction, dtype=dtm.DATE):
    """N-ary (start, end)."""

    __slots__ = ("interval", "closed")  # noqa: RUF023
    interval: int
    closed: ClosedInterval


class LinearSpace(RangeFunction, dtype=map_all(dtm.floats_dtype)):
    """N-ary (start, end)."""

    __slots__ = ("num_samples", "closed")  # noqa: RUF023
    num_samples: int
    closed: ClosedInterval

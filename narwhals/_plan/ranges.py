from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR, Function
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expr import RangeExpr
    from narwhals.dtypes import IntegerType


class RangeFunction(Function):
    def __repr__(self) -> str:
        tp = type(self)
        if tp is RangeFunction:
            return tp.__name__
        m: dict[type[RangeFunction], str] = {IntRange: "int_range"}
        return m[tp]

    def to_function_expr(self, *inputs: ExprIR) -> RangeExpr[Self]:
        from narwhals._plan.expr import RangeExpr

        return RangeExpr(input=inputs, function=self, options=self.function_options)


class IntRange(RangeFunction):
    """N-ary (start, end).

    Not implemented yet, but might push forward [#2722].

    See [`rust` entrypoint], which is roughly:

        Expr::Function { [start, end], FunctionExpr::Range(RangeFunction::IntRange { step, dtype }) }

    `narwhals` equivalent:

        FunctionExpr(input=(start, end), function=IntRange(step=step, dtype=dtype))

    [#2722]: https://github.com/narwhals-dev/narwhals/issues/2722
    [`rust` entrypoint]: https://github.com/pola-rs/polars/blob/2c7a3e77f0faa37c86a3745db4ef7707ae50c72e/crates/polars-plan/src/dsl/functions/range.rs#L14-L23
    """

    __slots__ = ("step", "dtype")  # noqa: RUF023
    step: int
    dtype: IntegerType

    @property
    def function_options(self) -> FunctionOptions:
        return FunctionOptions.row_separable()

    def unwrap_input(self, node: RangeExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        start, end = node.input
        return start, end

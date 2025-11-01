from __future__ import annotations

import operator as op
from typing import TYPE_CHECKING

from narwhals._plan._guards import is_function_expr
from narwhals._plan._immutable import Immutable
from narwhals._plan.exceptions import (
    binary_expr_length_changing_error,
    binary_expr_multi_output_error,
    binary_expr_shape_error,
)

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.expressions import BinaryExpr, BinarySelector, ExprIR
    from narwhals._plan.typing import (
        LeftSelectorT,
        LeftT,
        OperatorFn,
        RightSelectorT,
        RightT,
    )


class Operator(Immutable):
    _func: ClassVar[OperatorFn]
    _symbol: ClassVar[str]

    def __repr__(self) -> str:
        return self._symbol

    def __init_subclass__(
        cls, *args: Any, func: OperatorFn | None, symbol: str = "", **kwds: Any
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if func:
            cls._func = func
        cls._symbol = symbol or cls.__name__

    def to_binary_expr(
        self, left: LeftT, right: RightT, /
    ) -> BinaryExpr[LeftT, Self, RightT]:
        from narwhals._plan.expressions.expr import BinaryExpr

        if right.meta.has_multiple_outputs():
            raise binary_expr_multi_output_error(left, self, right)
        if _is_filtration(left):
            if _is_filtration(right):
                raise binary_expr_length_changing_error(left, self, right)
            if not right.is_scalar:
                raise binary_expr_shape_error(left, self, right)
        elif _is_filtration(right):
            if not left.is_scalar:
                raise binary_expr_shape_error(left, self, right)
        return BinaryExpr(left=left, op=self, right=right)

    def __call__(self, lhs: Any, rhs: Any) -> Any:
        """Apply binary operator to `left`, `right` operands."""
        return self.__class__._func(lhs, rhs)


def _is_filtration(ir: ExprIR) -> bool:
    return not ir.is_scalar and is_function_expr(ir) and not ir.options.is_elementwise()


class SelectorOperator(Operator, func=None):
    """Operators that can *also* be used in selectors."""

    def to_binary_selector(
        self, left: LeftSelectorT, right: RightSelectorT, /
    ) -> BinarySelector[LeftSelectorT, Self, RightSelectorT]:
        from narwhals._plan.expressions.expr import BinarySelector

        return BinarySelector(left=left, op=self, right=right)


# fmt: off
class Eq(Operator, func=op.eq, symbol="=="): ...
class NotEq(Operator, func=op.ne, symbol="!="): ...
class Lt(Operator, func=op.le, symbol="<"): ...
class LtEq(Operator, func=op.lt, symbol="<="): ...
class Gt(Operator, func=op.gt, symbol=">"): ...
class GtEq(Operator, func=op.ge, symbol=">="): ...
class Add(Operator, func=op.add, symbol="+"): ...
class Sub(SelectorOperator, func=op.sub, symbol="-"): ...
class Multiply(Operator, func=op.mul, symbol="*"): ...
class TrueDivide(Operator, func=op.truediv, symbol="/"): ...
class FloorDivide(Operator, func=op.floordiv, symbol="//"): ...
class Modulus(Operator, func=op.mod, symbol="%"): ...
class And(SelectorOperator, func=op.and_, symbol="&"): ...
class Or(SelectorOperator, func=op.or_, symbol="|"): ...
class ExclusiveOr(SelectorOperator, func=op.xor, symbol="^"): ...
# fmt: on

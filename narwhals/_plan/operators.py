from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from narwhals._plan.common import Immutable
from narwhals._plan.exceptions import (
    binary_expr_length_changing_error,
    binary_expr_multi_output_error,
    binary_expr_shape_error,
)
from narwhals._plan.expr import BinarySelector, FunctionExpr

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.expr import BinaryExpr, BinarySelector
    from narwhals._plan.typing import (
        LeftSelectorT,
        LeftT,
        OperatorFn,
        RightSelectorT,
        RightT,
    )


class Operator(Immutable):
    _op: ClassVar[OperatorFn]

    def __repr__(self) -> str:
        tp = type(self)
        if tp in {Operator, SelectorOperator}:
            return tp.__name__
        m = {
            Eq: "==",
            NotEq: "!=",
            Lt: "<",
            LtEq: "<=",
            Gt: ">",
            GtEq: ">=",
            Add: "+",
            Sub: "-",
            Multiply: "*",
            TrueDivide: "/",
            FloorDivide: "//",
            Modulus: "%",
            And: "&",
            Or: "|",
            ExclusiveOr: "^",
        }
        return m[tp]

    def to_binary_expr(
        self, left: LeftT, right: RightT, /
    ) -> BinaryExpr[LeftT, Self, RightT]:
        from narwhals._plan.expr import BinaryExpr

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
        return self.__class__._op(lhs, rhs)


def _is_filtration(ir: ExprIR) -> bool:
    if not ir.is_scalar and isinstance(ir, FunctionExpr):
        return not ir.options.is_elementwise()
    return False


class SelectorOperator(Operator):
    """Operators that can *also* be used in selectors.

    Remember that `Or` is named [`meta._selector_add`]!

    [`meta._selector_add`]: https://github.com/pola-rs/polars/blob/b9dd8cdbd6e6ec8373110536955ed5940b9460ec/crates/polars-plan/src/dsl/meta.rs#L113-L124
    """

    def to_binary_selector(
        self, left: LeftSelectorT, right: RightSelectorT, /
    ) -> BinarySelector[LeftSelectorT, Self, RightSelectorT]:
        from narwhals._plan.expr import BinarySelector

        return BinarySelector(left=left, op=self, right=right)


class Eq(Operator):
    _op = operator.eq


class NotEq(Operator):
    _op = operator.ne


class Lt(Operator):
    _op = operator.le


class LtEq(Operator):
    _op = operator.lt


class Gt(Operator):
    _op = operator.gt


class GtEq(Operator):
    _op = operator.ge


class Add(Operator):
    _op = operator.add


class Sub(SelectorOperator):
    _op = operator.sub


class Multiply(Operator):
    _op = operator.mul


class TrueDivide(Operator):
    _op = operator.truediv


class FloorDivide(Operator):
    _op = operator.floordiv


class Modulus(Operator):
    _op = operator.mod


class And(SelectorOperator):
    _op = operator.and_


class Or(SelectorOperator):
    _op = operator.or_


class ExclusiveOr(SelectorOperator):
    _op = operator.xor

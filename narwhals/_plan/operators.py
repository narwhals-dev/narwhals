from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR, Immutable
from narwhals._plan.expr import BinarySelector, FunctionExpr
from narwhals.exceptions import LengthChangingExprError, MultiOutputExpressionError

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self

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
            lhs_op = f"{left!r} {self!r} "
            rhs = repr(right)
            indent = len(lhs_op) * " "
            underline = len(rhs) * "^"
            msg = (
                "Multi-output expressions are only supported on the "
                f"left-hand side of a binary operation.\n"
                f"{lhs_op}{rhs}\n{indent}{underline}"
            )
            raise MultiOutputExpressionError(msg)

        if not any(_is_not_filtration(e) for e in (left, right)):
            lhs, rhs = repr(left), repr(right)
            op = f" {self!r} "
            underline_left = len(lhs) * "^"
            underline_right = len(rhs) * "^"
            pad_middle = len(op) * " "
            msg = (
                "Length-changing expressions can only be used in isolation, "
                "or followed by an aggregation.\n"
                f"{lhs}{op}{rhs}\n{underline_left}{pad_middle}{underline_right}"
            )
            raise LengthChangingExprError(msg)

        return BinaryExpr(left=left, op=self, right=right)

    def __call__(self, lhs: Any, rhs: Any) -> Any:
        """Apply binary operator to `left`, `right` operands."""
        return self.__class__._op(lhs, rhs)


def _is_not_filtration(ir: ExprIR) -> bool:
    # NOTE: Strange naming/negation is to short-circuit on the `any`
    if not ir.is_scalar and isinstance(ir, FunctionExpr):
        return ir.options.is_elementwise()
    return True


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

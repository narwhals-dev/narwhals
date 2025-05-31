from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIR, Immutable
from narwhals._plan.expr import BinarySelector, FunctionExpr
from narwhals.exceptions import (
    LengthChangingExprError,
    MultiOutputExpressionError,
    ShapeError,
)

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
            raise _bin_op_multi_output_error(left, self, right)

        if _is_filtration(left):
            if _is_filtration(right):
                raise _bin_op_length_changing_error(left, self, right)
            if not right.is_scalar:
                raise _bin_op_shape_error(left, self, right)
        elif _is_filtration(right):
            if not left.is_scalar:
                raise _bin_op_shape_error(left, self, right)

        return BinaryExpr(left=left, op=self, right=right)

    def __call__(self, lhs: Any, rhs: Any) -> Any:
        """Apply binary operator to `left`, `right` operands."""
        return self.__class__._op(lhs, rhs)


# NOTE: Always underlining `right`, since the message refers to both types of exprs
# Assuming the most recent as the issue
def _bin_op_shape_error(left: ExprIR, op: Operator, right: ExprIR) -> ShapeError:
    lhs_op = f"{left!r} {op!r} "
    rhs = repr(right)
    indent = len(lhs_op) * " "
    underline = len(rhs) * "^"
    msg = (
        f"Cannot combine length-changing expressions with length-preserving ones.\n"
        f"{lhs_op}{rhs}\n{indent}{underline}"
    )
    return ShapeError(msg)


def _bin_op_multi_output_error(
    left: ExprIR, op: Operator, right: ExprIR
) -> MultiOutputExpressionError:
    lhs_op = f"{left!r} {op!r} "
    rhs = repr(right)
    indent = len(lhs_op) * " "
    underline = len(rhs) * "^"
    msg = (
        "Multi-output expressions are only supported on the "
        f"left-hand side of a binary operation.\n"
        f"{lhs_op}{rhs}\n{indent}{underline}"
    )
    return MultiOutputExpressionError(msg)


def _bin_op_length_changing_error(
    left: ExprIR, op: Operator, right: ExprIR
) -> LengthChangingExprError:
    lhs, rhs = repr(left), repr(right)
    op_s = f" {op!r} "
    underline_left = len(lhs) * "^"
    underline_right = len(rhs) * "^"
    pad_middle = len(op_s) * " "
    msg = (
        "Length-changing expressions can only be used in isolation, "
        "or followed by an aggregation.\n"
        f"{lhs}{op_s}{rhs}\n{underline_left}{pad_middle}{underline_right}"
    )
    return LengthChangingExprError(msg)


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

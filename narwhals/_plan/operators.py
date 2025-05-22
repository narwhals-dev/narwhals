from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expr import BinaryExpr
    from narwhals._plan.typing import LeftT, RightT

from narwhals._plan.common import Immutable


class Operator(Immutable):
    def __repr__(self) -> str:
        tp = type(self)
        if tp is Operator:
            return "Operator"
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

        return BinaryExpr(left=left, op=self, right=right)


class Eq(Operator): ...


class NotEq(Operator): ...


class Lt(Operator): ...


class LtEq(Operator): ...


class Gt(Operator): ...


class GtEq(Operator): ...


class Add(Operator): ...


class Sub(Operator): ...


class Multiply(Operator): ...


class TrueDivide(Operator): ...


class FloorDivide(Operator): ...


class Modulus(Operator): ...


class And(Operator): ...


class Or(Operator): ...


class ExclusiveOr(Operator): ...

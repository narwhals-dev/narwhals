from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expr import BinaryExpr, BinarySelector, SelectorIR
    from narwhals._plan.typing import LeftT, RightT

from narwhals._plan.common import Immutable


class Operator(Immutable):
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

        return BinaryExpr(left=left, op=self, right=right)


class SelectorOperator(Operator):
    """Operators that can *also* be used in selectors.

    Remember that `Or` is named [`meta._selector_add`]!

    [`meta._selector_add`]: https://github.com/pola-rs/polars/blob/b9dd8cdbd6e6ec8373110536955ed5940b9460ec/crates/polars-plan/src/dsl/meta.rs#L113-L124
    """

    def to_binary_selector(
        self, left: SelectorIR, right: SelectorIR, /
    ) -> BinarySelector[Self]:
        from narwhals._plan.expr import BinarySelector

        return BinarySelector(left=left, op=self, right=right)


class Eq(Operator): ...


class NotEq(Operator): ...


class Lt(Operator): ...


class LtEq(Operator): ...


class Gt(Operator): ...


class GtEq(Operator): ...


class Add(Operator): ...


class Sub(SelectorOperator): ...


class Multiply(Operator): ...


class TrueDivide(Operator): ...


class FloorDivide(Operator): ...


class Modulus(Operator): ...


class And(SelectorOperator): ...


class Or(SelectorOperator): ...


class ExclusiveOr(SelectorOperator): ...

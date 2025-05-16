from __future__ import annotations

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
        }
        return m[tp]


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

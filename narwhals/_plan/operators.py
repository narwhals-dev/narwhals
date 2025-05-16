from __future__ import annotations

from narwhals._plan.common import Immutable


class Operator(Immutable): ...


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


class Not(Operator):
    """`__invert__`."""

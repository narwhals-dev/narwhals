from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import cast_for_truediv, floordiv_compat as floordiv

if TYPE_CHECKING:
    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        BinaryFunction,
        BinaryNumericTemporal,
        NumericScalar,
        UnaryNumeric,
    )

__all__ = ["add", "floordiv", "modulus", "multiply", "power", "sqrt", "sub", "truediv"]

add = t.cast("BinaryNumericTemporal", pc.add)
sub = t.cast("BinaryNumericTemporal", pc.subtract)
multiply = pc.multiply
power = t.cast("BinaryFunction[NumericScalar, NumericScalar]", pc.power)
sqrt = t.cast("UnaryNumeric", pc.sqrt)


def truediv(lhs: Incomplete, rhs: Incomplete) -> Incomplete:
    return pc.divide(*cast_for_truediv(lhs, rhs))


def modulus(lhs: Incomplete, rhs: Incomplete) -> Incomplete:
    floor_div = floordiv(lhs, rhs)
    return sub(lhs, multiply(floor_div, rhs))

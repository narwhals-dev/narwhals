from __future__ import annotations

import math
import typing as t
from typing import TYPE_CHECKING

import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import floordiv_compat as _floordiv
from narwhals._plan.arrow.functions._dtypes import F64, is_integer
from narwhals._plan.arrow.functions.meta import call

if TYPE_CHECKING:
    from narwhals._plan.arrow.typing import (
        BinaryFunction,
        BinaryNumericTemporal,
        ChunkedOrScalarAny,
        NumericScalar,
        UnaryNumeric,
    )

__all__ = [
    "abs",
    "add",
    "exp",
    "floordiv",
    "log",
    "modulus",
    "multiply",
    "power",
    "sqrt",
    "sub",
    "truediv",
]

add = t.cast("BinaryNumericTemporal", pc.add)
"""Equivalent to `lhs + rhs`."""
sub = t.cast("BinaryNumericTemporal", pc.subtract)
"""Equivalent to `lhs - rhs`."""
multiply = t.cast("BinaryNumericTemporal", pc.multiply)
"""Equivalent to `lhs * rhs`."""
floordiv = t.cast("BinaryNumericTemporal", _floordiv)
"""Equivalent to `lhs // rhs`."""
power = t.cast("BinaryFunction[NumericScalar, NumericScalar]", pc.power)
"""Equivalent to `lhs ** rhs`."""
sqrt = t.cast("UnaryNumeric", pc.sqrt)
"""Compute the square root of the elements."""
abs = t.cast("UnaryNumeric", pc.abs)
"""Compute absolute values."""
exp = t.cast("UnaryNumeric", pc.exp)
"""Compute the exponential, element-wise."""


def truediv(lhs: ChunkedOrScalarAny, rhs: ChunkedOrScalarAny, /) -> ChunkedOrScalarAny:
    """Equivalent to `lhs / rhs`."""
    if is_integer(lhs.type) and is_integer(rhs.type):
        lhs, rhs = lhs.cast(F64, safe=False), rhs.cast(F64, safe=False)
    result: ChunkedOrScalarAny = call("divide", lhs, rhs)
    return result


def modulus(lhs: ChunkedOrScalarAny, rhs: ChunkedOrScalarAny, /) -> ChunkedOrScalarAny:
    """Equivalent to `lhs % rhs`."""
    result: ChunkedOrScalarAny = sub(lhs, multiply(floordiv(lhs, rhs), rhs))
    return result


def log(native: ChunkedOrScalarAny, base: float = math.e) -> ChunkedOrScalarAny:
    """Compute the logarithm to a given base."""
    result: ChunkedOrScalarAny = call("logb", native, base)
    return result

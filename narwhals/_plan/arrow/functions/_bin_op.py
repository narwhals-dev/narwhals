from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import cast_for_truediv, floordiv_compat as _floordiv
from narwhals._plan.expressions import operators as ops

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        BinaryComp,
        BinaryFunction,
        BinaryLogical,
        BinaryNumericTemporal,
        BinOp,
        ChunkedOrScalarAny,
        NumericScalar,
    )

__all__ = [
    "add",
    "and_",
    "binary",
    "eq",
    "floordiv",
    "gt",
    "gt_eq",
    "lt",
    "lt_eq",
    "modulus",
    "multiply",
    "not_eq",
    "or_",
    "power",
    "sub",
    "truediv",
    "xor",
]

and_ = t.cast("BinaryLogical", pc.and_kleene)
or_ = t.cast("BinaryLogical", pc.or_kleene)
xor = t.cast("BinaryLogical", pc.xor)

eq = t.cast("BinaryComp", pc.equal)
not_eq = t.cast("BinaryComp", pc.not_equal)
gt_eq = t.cast("BinaryComp", pc.greater_equal)
gt = t.cast("BinaryComp", pc.greater)
lt_eq = t.cast("BinaryComp", pc.less_equal)
lt = t.cast("BinaryComp", pc.less)

add = t.cast("BinaryNumericTemporal", pc.add)
sub = t.cast("BinaryNumericTemporal", pc.subtract)
multiply = pc.multiply
floordiv = _floordiv
power = t.cast("BinaryFunction[NumericScalar, NumericScalar]", pc.power)


def truediv(lhs: Incomplete, rhs: Incomplete) -> Incomplete:
    return pc.divide(*cast_for_truediv(lhs, rhs))


def modulus(lhs: Incomplete, rhs: Incomplete) -> Incomplete:
    floor_div = floordiv(lhs, rhs)
    return sub(lhs, multiply(floor_div, rhs))


def binary(
    lhs: ChunkedOrScalarAny, op: type[ops.Operator], rhs: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return _DISPATCH_BINARY[op](lhs, rhs)


# TODO @dangotbanned: Somehow fix the typing on this
# - `_ArrowDispatch` is relying on the gradual typing
_DISPATCH_BINARY: Mapping[type[ops.Operator], BinOp] = {
    # BinaryComp
    ops.Eq: eq,
    ops.NotEq: not_eq,
    ops.Lt: lt,
    ops.LtEq: lt_eq,
    ops.Gt: gt,
    ops.GtEq: gt_eq,
    # BinaryFunction (well it should be)
    ops.Add: add,  # BinaryNumericTemporal
    ops.Sub: sub,  # pyarrow-stubs
    ops.Multiply: multiply,  # pyarrow-stubs
    ops.TrueDivide: truediv,  # [[Any, Any], Any]
    ops.FloorDivide: floordiv,  # [[ArrayOrScalar, ArrayOrScalar], Any]
    ops.Modulus: modulus,  # [[Any, Any], Any]
    # BinaryLogical
    ops.And: and_,
    ops.Or: or_,
    ops.ExclusiveOr: xor,
}

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow.functions import _arithmetic as arith
from narwhals._plan.expressions import operators as ops

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan.arrow.typing import (
        BinaryComp,
        BinaryLogical,
        BinOp,
        ChunkedOrScalarAny,
    )

__all__ = ["and_", "binary", "eq", "gt", "gt_eq", "lt", "lt_eq", "not_eq", "or_", "xor"]

and_ = t.cast("BinaryLogical", pc.and_kleene)
"""Equivalent to `lhs & rhs`."""
or_ = t.cast("BinaryLogical", pc.or_kleene)
"""Equivalent to `lhs | rhs`."""
xor = t.cast("BinaryLogical", pc.xor)
"""Equivalent to `lhs ^ rhs`."""

eq = t.cast("BinaryComp", pc.equal)
"""Equivalent to `lhs == rhs`."""
not_eq = t.cast("BinaryComp", pc.not_equal)
"""Equivalent to `lhs != rhs`."""
gt_eq = t.cast("BinaryComp", pc.greater_equal)
"""Equivalent to `lhs >= rhs`."""
gt = t.cast("BinaryComp", pc.greater)
"""Equivalent to `lhs > rhs`."""
lt_eq = t.cast("BinaryComp", pc.less_equal)
"""Equivalent to `lhs <= rhs`."""
lt = t.cast("BinaryComp", pc.less)
"""Equivalent to `lhs < rhs`."""


def binary(
    lhs: ChunkedOrScalarAny, op: type[ops.Operator], rhs: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    """Dispatch a binary operator type to a native function, providing `lhs` and `rhs` as operands."""
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
    ops.Add: arith.add,  # BinaryNumericTemporal
    ops.Sub: arith.sub,  # BinaryNumericTemporal
    ops.Multiply: arith.multiply,  # BinaryNumericTemporal
    ops.TrueDivide: arith.truediv,  # [[ChunkedOrScalarAny, ChunkedOrScalarAny], ChunkedOrScalarAny]
    ops.FloorDivide: arith.floordiv,  # BinaryNumericTemporal
    ops.Modulus: arith.modulus,  # [[ChunkedOrScalarAny, ChunkedOrScalarAny], ChunkedOrScalarAny]
    # BinaryLogical
    ops.And: and_,
    ops.Or: or_,
    ops.ExclusiveOr: xor,
}

from __future__ import annotations

from typing import Literal, TypeAlias

from narwhals._plan import expressions as ir

UnsupportedKind: TypeAlias = Literal[
    "vega expression", "window transform", "aggregate transform"
]


def unsupported_error(expr: ir.ExprIR, /, kind: UnsupportedKind) -> NotImplementedError:
    if isinstance(expr, ir.FunctionExpr):
        name = type(expr.function).__name__
    else:
        name = type(expr).__name__
    msg = f"Converting {name!r} into a {kind} is not yet implemented, got: {expr!r}"
    return NotImplementedError(msg)

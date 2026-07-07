from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from narwhals._plan import expressions as ir

if TYPE_CHECKING:
    from collections.abc import Mapping


Target: TypeAlias = Literal["expr", "window", "aggregate", "encoding"]
_TARGET_MSG: Mapping[Target, str] = {
    "expr": "vega expression",
    "window": "window transform",
    "aggregate": "aggregate transform",
    "encoding": "encoding",
}
_REASON_MSG: Mapping[str | None, str] = {
    "non-default": "(with non-default arguments)",
    None: "",
}


def unsupported_error(
    expr: ir.ExprIR, /, target: Target, reason: Literal["non-default"] | None = None
) -> NotImplementedError:
    if isinstance(expr, ir.FunctionExpr):
        name = type(expr.function).__name__
    else:
        name = type(expr).__name__
    what = f"Converting {name!r} into a {_TARGET_MSG[target]}"
    not_impl = f"is not yet implemented, got:\n    {expr!r}"
    args = (what, why, not_impl) if (why := _REASON_MSG[reason]) else (what, not_impl)
    msg = " ".join(args)
    return NotImplementedError(msg)

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._guards import is_expr
from narwhals._plan.common import ExprIR, NamedIR

if TYPE_CHECKING:
    from typing_extensions import LiteralString

    from narwhals._plan.dummy import Expr


def _unwrap_ir(obj: Expr | ExprIR | NamedIR) -> ExprIR:
    if is_expr(obj):
        return obj._ir
    if isinstance(obj, ExprIR):
        return obj
    if isinstance(obj, NamedIR):
        return obj.expr
    raise NotImplementedError(type(obj))


def assert_expr_ir_equal(
    actual: Expr | ExprIR | NamedIR, expected: Expr | ExprIR | NamedIR | LiteralString, /
) -> None:
    """Assert that `actual` is equivalent to `expected`.

    Arguments:
        actual: Result expression or IR to compare.
        expected: Target expression, IR, or repr to compare.

    Notes:
        Performing a repr comparison is more fragile, so should be avoided
        *unless* we raise an error at creation time.
    """
    lhs = _unwrap_ir(actual)
    if isinstance(expected, str):
        assert repr(lhs) == expected
    elif isinstance(actual, NamedIR) and isinstance(expected, NamedIR):
        assert actual == expected
    else:
        rhs = expected._ir if is_expr(expected) else expected
        assert lhs == rhs

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals import _plan as nwp
from narwhals._plan import expressions as ir
from narwhals._plan.common import NamedIR

if TYPE_CHECKING:
    from typing_extensions import LiteralString


def _unwrap_ir(obj: nwp.Expr | ir.ExprIR | NamedIR) -> ir.ExprIR:
    if isinstance(obj, nwp.Expr):
        return obj._ir
    if isinstance(obj, ir.ExprIR):
        return obj
    if isinstance(obj, NamedIR):
        return obj.expr
    raise NotImplementedError(type(obj))


def assert_expr_ir_equal(
    actual: nwp.Expr | ir.ExprIR | NamedIR,
    expected: nwp.Expr | ir.ExprIR | NamedIR | LiteralString,
    /,
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
        rhs = expected._ir if isinstance(expected, nwp.Expr) else expected
        assert lhs == rhs

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import is_expr

if TYPE_CHECKING:
    from typing_extensions import LiteralString

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummyExpr


def assert_expr_ir_equal(
    actual: DummyExpr | ExprIR, expected: DummyExpr | ExprIR | LiteralString, /
) -> None:
    """Assert that `actual` is equivalent to `expected`.

    Arguments:
        actual: Result expression or IR to compare.
        expected: Target expression, IR, or repr to compare.

    Notes:
        Performing a repr comparison is more fragile, so should be avoided
        *unless* we raise an error at creation time.
    """
    lhs = actual._ir if is_expr(actual) else actual
    if isinstance(expected, str):
        assert repr(lhs) == expected
    else:
        rhs = expected._ir if is_expr(expected) else expected
        assert lhs == rhs

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import is_expr

if TYPE_CHECKING:
    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummyExpr


def assert_expr_ir_equal(left: DummyExpr | ExprIR, right: DummyExpr | ExprIR) -> None:
    lhs = left._ir if is_expr(left) else left
    rhs = right._ir if is_expr(right) else right
    assert lhs == rhs

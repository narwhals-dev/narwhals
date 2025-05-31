from __future__ import annotations

import typing as t

from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant import CompliantNamespace as Namespace
    from narwhals._compliant.typing import CompliantExprAny
    from narwhals._plan import operators as ops
    from narwhals._plan.common import ExprIR, Function, IRNamespace, SelectorIR
    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.functions import RollingWindow
    from narwhals.typing import NonNestedLiteral

__all__ = ["FunctionT", "LeftT", "OperatorT", "RightT", "RollingT", "SelectorOperatorT"]


FunctionT = TypeVar("FunctionT", bound="Function")
RollingT = TypeVar("RollingT", bound="RollingWindow")
LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
OperatorT = TypeVar("OperatorT", bound="ops.Operator", default="ops.Operator")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")
OperatorFn: TypeAlias = "t.Callable[[t.Any, t.Any], t.Any]"

LeftSelectorT = TypeVar("LeftSelectorT", bound="SelectorIR", default="SelectorIR")
RightSelectorT = TypeVar("RightSelectorT", bound="SelectorIR", default="SelectorIR")
SelectorOperatorT = TypeVar(
    "SelectorOperatorT", bound="ops.SelectorOperator", default="ops.SelectorOperator"
)
IRNamespaceT = TypeVar("IRNamespaceT", bound="IRNamespace")

NonNestedLiteralT = TypeVar(
    "NonNestedLiteralT", bound="NonNestedLiteral", default="NonNestedLiteral"
)
LiteralT = TypeVar("LiteralT", bound="NonNestedLiteral | DummySeries", default=t.Any)

# NOTE: Shorter aliases of `_compliant.typing`
# - Aiming to try and preserve the types as much as possible
# - Recursion between `Expr` and `Frame` is an issue
Expr: TypeAlias = "CompliantExprAny"
ExprT = TypeVar("ExprT", bound="Expr")
Ns: TypeAlias = "Namespace[t.Any, ExprT]"
"""A `CompliantNamespace`, ignoring the `Frame` type."""

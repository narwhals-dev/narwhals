from __future__ import annotations

import typing as t

from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant import CompliantNamespace as Namespace
    from narwhals._compliant.typing import CompliantExprAny
    from narwhals._plan import operators as ops
    from narwhals._plan.common import ExprIR, Function, IRNamespace, SelectorIR
    from narwhals._plan.dummy import DummyExpr, DummySeries
    from narwhals._plan.functions import RollingWindow
    from narwhals.typing import NonNestedLiteral

__all__ = [
    "FunctionT",
    "IntoExpr",
    "IntoExprColumn",
    "LeftSelectorT",
    "LeftT",
    "LiteralT",
    "MapIR",
    "NonNestedLiteralT",
    "OperatorFn",
    "OperatorT",
    "RightSelectorT",
    "RightT",
    "RollingT",
    "SelectorOperatorT",
    "SelectorT",
    "Seq",
    "Udf",
]


FunctionT = TypeVar("FunctionT", bound="Function")
RollingT = TypeVar("RollingT", bound="RollingWindow")
LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
LeftT2 = TypeVar("LeftT2", bound="ExprIR", default="ExprIR")
OperatorT = TypeVar("OperatorT", bound="ops.Operator", default="ops.Operator")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")
RightT2 = TypeVar("RightT2", bound="ExprIR", default="ExprIR")
OperatorFn: TypeAlias = "t.Callable[[t.Any, t.Any], t.Any]"
ExprIRT = TypeVar("ExprIRT", bound="ExprIR", default="ExprIR")
ExprIRT2 = TypeVar("ExprIRT2", bound="ExprIR", default="ExprIR")

SelectorT = TypeVar("SelectorT", bound="SelectorIR", default="SelectorIR")
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
MapIR: TypeAlias = "t.Callable[[ExprIR], ExprIR]"
"""A function to apply to all nodes in this tree."""

# NOTE: Shorter aliases of `_compliant.typing`
# - Aiming to try and preserve the types as much as possible
# - Recursion between `Expr` and `Frame` is an issue
Expr: TypeAlias = "CompliantExprAny"
ExprT = TypeVar("ExprT", bound="Expr")
Ns: TypeAlias = "Namespace[t.Any, ExprT]"
"""A `CompliantNamespace`, ignoring the `Frame` type."""


T = TypeVar("T")

Seq: TypeAlias = "tuple[T,...]"
"""Immutable Sequence.

Using instead of `Sequence`, as a `list` can be passed there (can't break immutability promise).
"""

Udf: TypeAlias = "t.Callable[[t.Any], t.Any]"
"""Placeholder for `map_batches(function=...)`."""

IntoExprColumn: TypeAlias = "DummyExpr | DummySeries | str"
IntoExpr: TypeAlias = "NonNestedLiteral | IntoExprColumn"

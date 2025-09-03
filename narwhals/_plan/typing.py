from __future__ import annotations

import typing as t

from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals import dtypes
    from narwhals._plan import operators as ops
    from narwhals._plan.common import ExprIR, Function, IRNamespace, NamedIR, SelectorIR
    from narwhals._plan.dummy import Expr, Series
    from narwhals._plan.functions import RollingWindow
    from narwhals._plan.ranges import RangeFunction
    from narwhals.typing import (
        NativeDataFrame,
        NativeFrame,
        NativeSeries,
        NonNestedDType,
        NonNestedLiteral,
    )

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
    "RangeT",
    "RightSelectorT",
    "RightT",
    "RollingT",
    "SelectorOperatorT",
    "SelectorT",
    "Seq",
    "Udf",
]


FunctionT = TypeVar("FunctionT", bound="Function", default="Function")
RollingT = TypeVar("RollingT", bound="RollingWindow", default="RollingWindow")
RangeT = TypeVar("RangeT", bound="RangeFunction", default="RangeFunction")
LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
OperatorT = TypeVar("OperatorT", bound="ops.Operator", default="ops.Operator")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")
OperatorFn: TypeAlias = "t.Callable[[t.Any, t.Any], t.Any]"
ExprIRT = TypeVar("ExprIRT", bound="ExprIR", default="ExprIR")
ExprIRT2 = TypeVar("ExprIRT2", bound="ExprIR", default="ExprIR")
NamedOrExprIRT = TypeVar("NamedOrExprIRT", "NamedIR[t.Any]", "ExprIR")

SelectorT = TypeVar("SelectorT", bound="SelectorIR", default="SelectorIR")
LeftSelectorT = TypeVar("LeftSelectorT", bound="SelectorIR", default="SelectorIR")
RightSelectorT = TypeVar("RightSelectorT", bound="SelectorIR", default="SelectorIR")
SelectorOperatorT = TypeVar(
    "SelectorOperatorT", bound="ops.SelectorOperator", default="ops.SelectorOperator"
)
IRNamespaceT = TypeVar("IRNamespaceT", bound="IRNamespace")
Accessor: TypeAlias = t.Literal[
    "arr", "cat", "dt", "list", "meta", "name", "str", "bin", "struct"
]
"""Namespace accessor property name."""

DTypeT = TypeVar("DTypeT", bound="dtypes.DType")
NonNestedDTypeT = TypeVar("NonNestedDTypeT", bound="NonNestedDType")

NonNestedLiteralT = TypeVar(
    "NonNestedLiteralT", bound="NonNestedLiteral", default="NonNestedLiteral"
)
NativeSeriesT = TypeVar("NativeSeriesT", bound="NativeSeries", default="NativeSeries")
NativeFrameT = TypeVar("NativeFrameT", bound="NativeFrame", default="NativeFrame")
NativeDataFrameT = TypeVar(
    "NativeDataFrameT", bound="NativeDataFrame", default="NativeDataFrame"
)
LiteralT = TypeVar("LiteralT", bound="NonNestedLiteral | Series[t.Any]", default=t.Any)
MapIR: TypeAlias = "t.Callable[[ExprIR], ExprIR]"
"""A function to apply to all nodes in this tree."""

T = TypeVar("T")

Seq: TypeAlias = "tuple[T,...]"
"""Immutable Sequence.

Using instead of `Sequence`, as a `list` can be passed there (can't break immutability promise).
"""

Udf: TypeAlias = "t.Callable[[t.Any], t.Any]"
"""Placeholder for `map_batches(function=...)`."""

IntoExprColumn: TypeAlias = "Expr | Series[t.Any] | str"
IntoExpr: TypeAlias = "NonNestedLiteral | IntoExprColumn"

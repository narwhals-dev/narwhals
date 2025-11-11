from __future__ import annotations

import typing as t
from collections.abc import Container
from typing import TYPE_CHECKING

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from typing_extensions import TypeAlias

    from narwhals import dtypes
    from narwhals._native import NativeDataFrame, NativeFrame, NativeSeries
    from narwhals._plan._expr_ir import ExprIR, NamedIR, SelectorIR
    from narwhals._plan._function import Function
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import operators as ops
    from narwhals._plan.expressions.functions import RollingWindow
    from narwhals._plan.expressions.namespace import IRNamespace
    from narwhals._plan.expressions.ranges import RangeFunction
    from narwhals._plan.selectors import Selector
    from narwhals._plan.series import Series
    from narwhals.typing import NonNestedDType, NonNestedLiteral

__all__ = [
    "ColumnNameOrSelector",
    "DataFrameT",
    "FunctionT",
    "Ignored",
    "IntoExpr",
    "IntoExprColumn",
    "LeftSelectorT",
    "LeftT",
    "LiteralT",
    "MapIR",
    "NonNestedLiteralT",
    "OperatorFn",
    "OperatorT",
    "RangeT_co",
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
FunctionT_co = TypeVar(
    "FunctionT_co", bound="Function", default="Function", covariant=True
)
RollingT_co = TypeVar(
    "RollingT_co", bound="RollingWindow", default="RollingWindow", covariant=True
)
RangeT_co = TypeVar(
    "RangeT_co", bound="RangeFunction", default="RangeFunction", covariant=True
)
LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
OperatorT = TypeVar("OperatorT", bound="ops.Operator", default="ops.Operator")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")
OperatorFn: TypeAlias = "Callable[[t.Any, t.Any], t.Any]"
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
NativeSeriesAnyT = TypeVar("NativeSeriesAnyT", bound="NativeSeries", default="t.Any")
NativeSeriesT_co = TypeVar(
    "NativeSeriesT_co", bound="NativeSeries", covariant=True, default="NativeSeries"
)
NativeFrameT = TypeVar("NativeFrameT", bound="NativeFrame", default="NativeFrame")
NativeFrameT_co = TypeVar(
    "NativeFrameT_co", bound="NativeFrame", covariant=True, default="NativeFrame"
)
NativeDataFrameT = TypeVar(
    "NativeDataFrameT", bound="NativeDataFrame", default="NativeDataFrame"
)
NativeDataFrameT_co = TypeVar(
    "NativeDataFrameT_co",
    bound="NativeDataFrame",
    covariant=True,
    default="NativeDataFrame",
)
LiteralT = TypeVar("LiteralT", bound="NonNestedLiteral | Series[t.Any]", default=t.Any)
MapIR: TypeAlias = "Callable[[ExprIR], ExprIR]"
"""A function to apply to all nodes in this tree."""

T = TypeVar("T")

Seq: TypeAlias = tuple[T, ...]
"""Immutable Sequence.

Using instead of `Sequence`, as a `list` can be passed there (can't break immutability promise).
"""

Udf: TypeAlias = "Callable[[t.Any], t.Any]"
"""Placeholder for `map_batches(function=...)`."""

IntoExprColumn: TypeAlias = "Expr | Series[t.Any] | str"
IntoExpr: TypeAlias = "NonNestedLiteral | IntoExprColumn"
ColumnNameOrSelector: TypeAlias = "str | Selector"
OneOrIterable: TypeAlias = "T | t.Iterable[T]"
OneOrSeq: TypeAlias = t.Union[T, Seq[T]]
DataFrameT = TypeVar("DataFrameT", bound="DataFrame[t.Any, t.Any]")
Order: TypeAlias = t.Literal["ascending", "descending"]
NonCrossJoinStrategy: TypeAlias = t.Literal["inner", "left", "full", "semi", "anti"]
PartialSeries: TypeAlias = "Callable[[Iterable[t.Any]], Series[NativeSeriesAnyT]]"


Ignored: TypeAlias = Container[str]
"""Names of `group_by` columns, which are excluded[^1] when expanding a `Selector`.

[^1]: `ByName`, `ByIndex` will never be ignored.
"""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Literal

from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable
    from types import MappingProxyType, ModuleType as _ModuleType

    from typing_extensions import LiteralString, TypeAlias

    from narwhals import dtypes
    from narwhals._native import (
        NativeDataFrame,
        NativeFrame,
        NativeLazyFrame,
        NativeSeries,
    )
    from narwhals._plan._expr_ir import ExprIR, NamedIR, SelectorIR
    from narwhals._plan._function import Function, HorizontalFunction
    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.compliant.plugins import Plugin
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import operators as ops
    from narwhals._plan.expressions.aggregation import AggExpr
    from narwhals._plan.expressions.namespace import IRNamespace
    from narwhals._plan.expressions.ranges import RangeFunction
    from narwhals._plan.expressions.struct import StructFunction
    from narwhals._plan.lazyframe import LazyFrame
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.selectors import Selector
    from narwhals._plan.series import Series
    from narwhals._typing import LazyOnly, PandasLike, _EagerAllowedImpl, _LazyAllowedImpl
    from narwhals._utils import Implementation
    from narwhals.typing import (
        Backend,
        IntoBackend,
        NonNestedDType,
        NonNestedLiteral,
        PythonLiteral,
    )

__all__ = (
    "AggExprT_co",
    "ColumnNameOrSelector",
    "DataFrameT",
    "FunctionT",
    "Ignored",
    "IntoExpr",
    "IntoExprColumn",
    "LeftSelectorT",
    "LeftT",
    "LiteralT_co",
    "MapIR",
    "NonNestedLiteralT_co",
    "OperatorFn",
    "OperatorT",
    "OutputNames",
    "RangeT_co",
    "RightSelectorT",
    "RightT",
    "SelectorOperatorT",
    "SelectorT",
    "Seq",
    "Udf",
)


FunctionT = TypeVar("FunctionT", bound="Function", default="Function")
FunctionT_co = TypeVar(
    "FunctionT_co", bound="Function", default="Function", covariant=True
)
RangeT_co = TypeVar(
    "RangeT_co",
    bound="RangeFunction[t.Any]",
    default="RangeFunction[t.Any]",
    covariant=True,
)
StructT_co = TypeVar(
    "StructT_co", bound="StructFunction", default="StructFunction", covariant=True
)
HorizontalT_co = TypeVar(
    "HorizontalT_co",
    bound="HorizontalFunction",
    default="HorizontalFunction",
    covariant=True,
)
LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")
LeftT_co = TypeVar("LeftT_co", bound="ExprIR", default="ExprIR", covariant=True)
RightT_co = TypeVar("RightT_co", bound="ExprIR", default="ExprIR", covariant=True)
OperatorT = TypeVar("OperatorT", bound="ops.Operator", default="ops.Operator")
OperatorFn: TypeAlias = "Callable[[t.Any, t.Any], t.Any]"
ExprIRT = TypeVar("ExprIRT", bound="ExprIR", default="ExprIR")
ExprIRT_co = TypeVar("ExprIRT_co", bound="ExprIR", default="ExprIR", covariant=True)
NamedOrExprIRT = TypeVar("NamedOrExprIRT", "NamedIR[t.Any]", "ExprIR")
AggExprT_co = TypeVar("AggExprT_co", bound="AggExpr", default="AggExpr", covariant=True)
SelectorT = TypeVar("SelectorT", bound="SelectorIR", default="SelectorIR")
LeftSelectorT = TypeVar("LeftSelectorT", bound="SelectorIR", default="SelectorIR")
RightSelectorT = TypeVar("RightSelectorT", bound="SelectorIR", default="SelectorIR")
SelectorT_co = TypeVar(
    "SelectorT_co", bound="SelectorIR", default="SelectorIR", covariant=True
)
LeftSelectorT_co = TypeVar(
    "LeftSelectorT_co", bound="SelectorIR", default="SelectorIR", covariant=True
)
RightSelectorT_co = TypeVar(
    "RightSelectorT_co", bound="SelectorIR", default="SelectorIR", covariant=True
)
SelectorOperatorT = TypeVar(
    "SelectorOperatorT", bound="ops.SelectorOperator", default="ops.SelectorOperator"
)
"""An `Operator` used in a `BinarySelector`.

One of:

    ┌─────────────┬────────────┬──────────────────────┐
    │ Operator    ┆ Expression ┆ set operation        │
    ╞═════════════╪════════════╪══════════════════════╡
    │ And         ┆ A & B      ┆ intersection         │
    │ Or          ┆ A | B      ┆ union                │
    │ Sub         ┆ A - B      ┆ difference           │
    │ ExclusiveOr ┆ A ^ B      ┆ symmetric_difference │
    └─────────────┴────────────┴──────────────────────┘
"""
IRNamespaceT = TypeVar("IRNamespaceT", bound="IRNamespace")
Accessor: TypeAlias = t.Literal[
    "arr", "cat", "dt", "list", "meta", "name", "str", "bin", "struct"
]
"""Namespace accessor property name."""

DTypeT = TypeVar("DTypeT", bound="dtypes.DType")
NonNestedDTypeT = TypeVar("NonNestedDTypeT", bound="NonNestedDType")

NonNestedLiteralT_co = TypeVar(
    "NonNestedLiteralT_co",
    bound="NonNestedLiteral",
    default="NonNestedLiteral",
    covariant=True,
)
PythonLiteralT = TypeVar("PythonLiteralT", bound="PythonLiteral", default="PythonLiteral")
PythonLiteralT_co = TypeVar(
    "PythonLiteralT_co", bound="PythonLiteral", covariant=True, default="PythonLiteral"
)
NativeSeriesT = TypeVar("NativeSeriesT", bound="NativeSeries", default="NativeSeries")
NativeSeriesAnyT = TypeVar("NativeSeriesAnyT", bound="NativeSeries", default="t.Any")
NativeSeriesT_co = TypeVar(
    "NativeSeriesT_co", bound="NativeSeries", covariant=True, default="NativeSeries"
)
NativeFrameT_co = TypeVar("NativeFrameT_co", bound="NativeFrame", covariant=True)
NativeDataFrameT = TypeVar(
    "NativeDataFrameT", bound="NativeDataFrame", default="NativeDataFrame"
)
NativeDataFrameT_co = TypeVar(
    "NativeDataFrameT_co",
    bound="NativeDataFrame",
    covariant=True,
    default="NativeDataFrame",
)
NativeLazyFrameT = TypeVar(
    "NativeLazyFrameT", bound="NativeLazyFrame", default="NativeLazyFrame"
)
NativeLazyFrameT_co = TypeVar(
    "NativeLazyFrameT_co",
    bound="NativeLazyFrame",
    default="NativeLazyFrame",
    covariant=True,
)
LiteralT_co = TypeVar(
    "LiteralT_co", bound="PythonLiteral | Series[t.Any]", covariant=True, default=t.Any
)
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
IntoExpr: TypeAlias = "PythonLiteral | IntoExprColumn"
ColumnNameOrSelector: TypeAlias = "str | Selector"
OneOrIterable: TypeAlias = "T | Iterable[T]"
OneOrSeq: TypeAlias = t.Union[T, Seq[T]]
DataFrameT = TypeVar("DataFrameT", bound="DataFrame[t.Any, t.Any]")
LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[t.Any]")
SeriesT = TypeVar("SeriesT", bound="Series[t.Any]")
Order: TypeAlias = t.Literal["ascending", "descending"]
NonCrossJoinStrategy: TypeAlias = t.Literal["inner", "left", "full", "semi", "anti"]
PartialSeries: TypeAlias = "Callable[[Iterable[t.Any]], Series[NativeSeriesAnyT]]"
ClosedKwds: TypeAlias = "Callable[[], MappingProxyType[str, t.Any]]"
"""A zero-argument callable that produces *closed-over* keyword arguments.

The return type of `closed_kwds`.
"""

OutputNames: TypeAlias = "Seq[str]"
"""Names of output columns after selectors expansion."""

Ignored: TypeAlias = "Container[str]"
"""Names of `group_by` key columns.

When expanding a selector, these columns will be excluded [^1] from the result.

[^1]: Except `ByName`, `ByIndex`.
"""


IncompleteCyclic: TypeAlias = "t.Any"
"""Placeholder for typing that introduces a cyclic dependency.

Mainly for spelling `(Compliant)DataFrame` from within `(Compliant)Series`.

On `main`, this works fine when running a type checker from the CLI - but causes
intermittent warnings when running in a language server.
"""


IncompleteVarianceLie: TypeAlias = "t.Any"
"""Placeholder for typing that would make a type parameter be [inferred] as invariant.

Escape hatch for protocols with `@classmethod`s that should be treated the same as `__init__`.

We need to define a constructor, but it can only be typed (and remain covariant),
if it is named `__init__`.

Defining `__init__` in a protocol is buggy, so `from_native` uses `Incomplete`.

[inferred]: https://typing.python.org/en/latest/spec/generics.html#variance
"""

KnownImpl: TypeAlias = "_EagerAllowedImpl | _LazyAllowedImpl"
"""Equivalent to `Backend - BackendName`."""


BackendTodo: TypeAlias = "PandasLike | LazyOnly"
"""Backends that are not *yet* implemented in `narwhals._plan`."""

NativeModuleType: TypeAlias = "_ModuleType"
"""*Represents* a strict subset of what is accepted by `Implementation.from_native_namespace(...)`.

The excluded modules are equivalent to the names excluded in `narwhals._plan.typing.BackendTodo`.

This [isn't representable in the current type system](https://github.com/python/typing/issues/1039)
"""

PluginName: TypeAlias = "LiteralString"
"""Name of a backend's [entry point].

This is ~~supported~~ planned to be supported wherever a `backend` parameter is requested.

## See Also
- [Using package metadata](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)
- [Entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/#data-model)

[entry point]: https://docs.python.org/3/library/importlib.metadata.html#importlib.metadata.EntryPoint
"""

IntoPlugin: TypeAlias = "IntoBackend[Backend] | PluginName | Implementation"
"""Anything that can be used to load a `Plugin`.

This is a superset of [`IntoBackend`], adding support for external plugin names.

Important:
    `Implementation.UNKNOWN` is not accepted at runtime and *eventually* the
    *opaque* `Implementation` should be removed from this definition.

[`IntoBackend`]: https://narwhals-dev.github.io/narwhals/api-reference/typing/#narwhals.typing.IntoBackend
"""

BuiltinAny: TypeAlias = "ArrowPlugin | PolarsPlugin"
"""An internal plugin (`Builtin`)."""

PluginAny: TypeAlias = "Plugin[t.Any, t.Any, t.Any, t.Any]"
"""An external plugin.

This type is assignable to any `Plugin`, but should be used *only* when we can't provide
anything more meaningful statically.

Tip:
    Prefer `BuiltinAny` whenever possible.
"""

VersionName: TypeAlias = Literal["MAIN", "V1", "V2"]

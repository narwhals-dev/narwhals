from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable, Container, Iterable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from narwhals._native import NativeDataFrame, NativeFrame, NativeSeries
from narwhals._typing import LazyOnly, PandasLike, _EagerAllowedImpl, _LazyAllowedImpl
from narwhals._typing_compat import TypeVar
from narwhals._utils import Implementation
from narwhals.typing import Backend, IntoBackend

if TYPE_CHECKING:
    from typing import TypeAlias

    from typing_extensions import LiteralString

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.compliant.plugins import Plugin
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expr import Expr
    from narwhals._plan.lazyframe import LazyFrame
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.selectors import Selector
    from narwhals._plan.series import Series
    from narwhals.typing import NonNestedLiteral, PythonLiteral

if sys.version_info >= (3, 14):
    from io import Writer
else:
    _T_contra = TypeVar("_T_contra", contravariant=True)

    class Writer(Protocol[_T_contra]):
        # https://docs.python.org/3/library/io.html#io.Writer
        __slots__ = ()

        def write(self, data: _T_contra, /) -> int: ...


if sys.version_info >= (3, 12):
    from collections.abc import Buffer as ReadableBuffer
else:

    class ReadableBuffer(Protocol):
        # https://github.com/python/typeshed/blob/abbf4372552d78cdd4514db2a2d855658c1a98a5/stdlib/_typeshed/__init__.pyi#L305-L306
        # https://github.com/python/typeshed/blob/abbf4372552d78cdd4514db2a2d855658c1a98a5/stdlib/_pickle.pyi#L41-L42
        # https://github.com/python/typeshed/blob/abbf4372552d78cdd4514db2a2d855658c1a98a5/stdlib/typing_extensions.pyi#L380-L387
        def __buffer__(self, flags: int, /) -> memoryview: ...


__all__ = (
    "ColumnNameOrSelector",
    "DataFrameT",
    "Ignored",
    "IntoExpr",
    "IntoExprColumn",
    "MapIR",
    "NonNestedLiteralT_co",
    "OutputNames",
    "Seq",
)


Accessor: TypeAlias = Literal[
    "arr", "cat", "dt", "list", "meta", "name", "str", "bin", "struct"
]
"""Namespace accessor property name."""
Constructs: TypeAlias = Literal["Expr", "Scalar"]
NonNestedLiteralT_co = TypeVar(
    "NonNestedLiteralT_co",
    bound="NonNestedLiteral",
    default="NonNestedLiteral",
    covariant=True,
)
NativeSeriesT = TypeVar("NativeSeriesT", bound=NativeSeries, default=NativeSeries)
NativeSeriesT_co = TypeVar(
    "NativeSeriesT_co", bound=NativeSeries, covariant=True, default=NativeSeries
)
NativeFrameT_co = TypeVar("NativeFrameT_co", bound=NativeFrame, covariant=True)
NativeDataFrameT = TypeVar(
    "NativeDataFrameT", bound=NativeDataFrame, default=NativeDataFrame
)
NativeDataFrameT_co = TypeVar(
    "NativeDataFrameT_co", bound=NativeDataFrame, covariant=True, default=NativeDataFrame
)
MapIR: TypeAlias = Callable[["ExprIR"], "ExprIR"]
"""A function to apply to all nodes in this tree."""

T = TypeVar("T")

Seq: TypeAlias = tuple[T, ...]
"""Immutable Sequence.

Using instead of `Sequence`, as a `list` can be passed there (can't break immutability promise).
"""

Seq1: TypeAlias = tuple[T]
Seq2: TypeAlias = tuple[T, T]
Seq3: TypeAlias = tuple[T, T, T]

IntoExprColumn: TypeAlias = "Expr | Series[Any] | str"
IntoExpr: TypeAlias = "PythonLiteral | IntoExprColumn"
ColumnNameOrSelector: TypeAlias = "str | Selector"
OneOrIterable: TypeAlias = T | Iterable[T]
OneOrSeq: TypeAlias = T | Seq[T]
DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any, Any]")
LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[Any]")
SeriesT = TypeVar("SeriesT", bound="Series[Any]")
NonCrossJoinStrategy: TypeAlias = Literal["inner", "left", "full", "semi", "anti"]
PartialSeries: TypeAlias = Callable[[Iterable[Any]], "Series[NativeSeriesT_co]"]
ClosedKwds: TypeAlias = Callable[[], types.MappingProxyType[str, Any]]
"""A zero-argument callable that produces *closed-over* keyword arguments.

The return type of `closed_kwds`.
"""

OutputNames: TypeAlias = Seq[str]
"""Names of output columns after selector expansion."""

Ignored: TypeAlias = Container[str]
"""Names of `group_by` key columns.

When expanding a selector, these columns will be excluded [^1] from the result.

[^1]: Except `ByName`, `ByIndex`.
"""

IncompleteCyclic: TypeAlias = Any
"""Placeholder for typing that introduces a cyclic dependency.

Mainly for spelling `(Compliant)DataFrame` from within `(Compliant)Series`.

On `main`, this works fine when running a type checker from the CLI - but causes
intermittent warnings when running in a language server.
"""


IncompleteVarianceLie: TypeAlias = Any
"""Placeholder for typing that would make a type parameter be [inferred] as invariant.

Escape hatch for protocols with `@classmethod`s that should be treated the same as `__init__`.

We need to define a constructor, but it can only be typed (and remain covariant),
if it is named `__init__`.

Defining `__init__` in a protocol is buggy, so `from_native` uses `Incomplete`.

[inferred]: https://typing.python.org/en/latest/spec/generics.html#variance
"""

KnownImpl: TypeAlias = _EagerAllowedImpl | _LazyAllowedImpl
"""Equivalent to `Backend - BackendName`."""


BackendTodo: TypeAlias = PandasLike | LazyOnly
"""Backends that are not *yet* implemented in `narwhals._plan`."""

NativeModuleType: TypeAlias = types.ModuleType
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

IntoPlugin: TypeAlias = IntoBackend[Backend] | PluginName | Implementation
"""Anything that can be used to load a `Plugin`.

This is a superset of [`narwhals.typing.IntoBackend`][], adding support for external plugin names.

Important:
    `Implementation.UNKNOWN` is not accepted at runtime and *eventually* the
    *opaque* `Implementation` should be removed from this definition.

[`narwhals.typing.IntoBackend`]: https://narwhals-dev.github.io/narwhals/api-reference/typing/#narwhals.typing.IntoBackend
"""

BuiltinAny: TypeAlias = "ArrowPlugin | PolarsPlugin"
"""An internal plugin (`Builtin`)."""

PluginAny: TypeAlias = "Plugin[Any, Any, Any, Any]"
"""An external plugin.

This type is assignable to any `Plugin`, but should be used *only* when we can't provide
anything more meaningful statically.

Tip:
    Prefer `BuiltinAny` whenever possible.
"""

VersionName: TypeAlias = Literal["MAIN", "V1", "V2"]


# `narwhals.typing.FileSource` uses a forward reference, which breaks cross-refs in the docs
FileSource: TypeAlias = str | os.PathLike[str]
"""Path to a file.

Either a string or an object that implements [`__fspath__`], such as [`pathlib.Path`].

[`__fspath__`]: https://docs.python.org/3/library/os.html#os.PathLike
[`pathlib.Path`]: https://docs.python.org/3/library/pathlib.html#pathlib.Path
"""


class PickleLoad(Protocol):
    """[`_typeshed._pickle._ReadableFileobj`](https://github.com/python/typeshed/blob/abbf4372552d78cdd4514db2a2d855658c1a98a5/stdlib/_pickle.pyi#L7-L10)."""

    def read(self, n: int, /) -> bytes: ...
    def readline(self) -> bytes: ...


SerdeFormat: TypeAlias = Literal["binary", "json"]
SerdeSource: TypeAlias = FileSource | ReadableBuffer | PickleLoad
"""Path to a file or a file-like object.

By file-like object, we refer to objects that have a `read()` method,
such as a file handler (e.g. via builtin [`open`][] function) or [`io.BytesIO`][]).
"""

SerdeSink: TypeAlias = FileSource | Writer[bytes]
"""File path to which the result should be written."""

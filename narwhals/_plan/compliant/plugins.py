"""An entrypoint for using a backend to implement narwhals-level operations.

## Notes
- All built-in backends must be implemented using the same machinery as extensions are expected to (`Plugin`)
- The `Plugin` protocol must be *satisfiable* by a package's `__init__.py`
    - But that should not prevent the *option* to implement using classes + inheritance
- All built-in backends must use lazy loading
    - This should be *encouraged* for extensions, but not required
This represents the **external-view** of the backend
    - The implementation can expose things to the `Plugin`, but must not depend on the plugin for implementation
"""

from __future__ import annotations

import functools
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, overload

from narwhals._plan._namespace import known_implementation
from narwhals._plan.compliant.classes import ClassesAny, ClassesT_co, HasClasses
from narwhals._plan.compliant.typing import (
    Native as LF,
    NativeDataFrameT as DF,
    NativeSeriesT as S,
)
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.metadata import EntryPoints

    from typing_extensions import LiteralString, Never, TypeAlias, TypeIs

    from narwhals._plan._namespace import KnownImpl
    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.typing import BackendTodo, NativeModuleType
    from narwhals._typing import Arrow, Polars
    from narwhals.typing import Backend, IntoBackend

    MYPY: Final = False
    if MYPY:
        # NOTE: Use this to avoid mypy's PEP 675 hack from blocking the feature
        # https://github.com/python/mypy/issues/12554
        LiteralString_: TypeAlias = Any
    else:
        from typing_extensions import LiteralString as LiteralString_


__all__ = ["Builtin", "Implementation", "Plugin", "PluginName", "Unsupported"]

PluginName: TypeAlias = "LiteralString"
"""Name of a backend's [entry point].

This is ~~supported~~ planned to be supported wherever a `backend` parameter is requested.

## See Also
- [Using package metadata](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)
- [Entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/#data-model)

[entry point]: https://docs.python.org/3/library/importlib.metadata.html#importlib.metadata.EntryPoint
"""

# NOTE: `Never` might be another option?
# try that out if `Any` causes *any* issues
Unsupported: TypeAlias = Any
"""Marker to use for types that are not planned to be implemented."""

PluginAny: TypeAlias = "Plugin[ClassesAny, Any, Any, Any]"
"""When used as a return type, this indicates an extension rather than a `Builtin`."""


BuiltinAny: TypeAlias = "ArrowPlugin | PolarsPlugin"
"""Backends defined inside of narwhals."""


class Plugin(HasClasses[ClassesT_co], Protocol[ClassesT_co, DF, LF, S]):
    """An entrypoint for using a backend to implement narwhals-level operations.

    `[ClassesT_co, DF, LF, S]`.
    """

    __slots__ = ()

    # TODO @dangotbanned: Think about how the narwhals-level will use this for state
    def is_loaded(self) -> bool: ...
    def is_available(self) -> bool: ...

    # TODO @dangotbanned: Do we still want to use these like in `narwhals._plan.translate.py`?
    def is_native(self, obj: Any, /) -> TypeIs[DF | LF | S]: ...
    def is_native_dataframe(self, obj: Any, /) -> TypeIs[DF]: ...
    def is_native_lazyframe(self, obj: Any, /) -> TypeIs[LF]: ...
    def is_native_series(self, obj: Any, /) -> TypeIs[S]: ...

    def native_dataframe_classes(self) -> Iterator[type[DF]]: ...
    def native_lazyframe_classes(self) -> Iterator[type[LF]]: ...
    def native_series_classes(self) -> Iterator[type[S]]: ...
    def native_classes(self) -> Iterator[type[DF | LF | S]]: ...

    # TODO @dangotbanned: Maybe remove since this information is available elsewhere?
    @property
    def plugin_name(self) -> PluginName: ...


# TODO @dangotbanned: Rename `Plugin.is_loaded` to reflect that it is about the native part
# TODO @dangotbanned: (low-priority) Preserve the exact `implementation` for each backend (bad overloads)
class Builtin(Plugin[ClassesT_co, DF, LF, S], Protocol[ClassesT_co, DF, LF, S]):
    """Backends defined inside of narwhals are plugins too.

    `[ClassesT_co, DF, LF, S]`.

    ## Notes
    - Might want to provide *parts* of this in a sub-protocol for `Plugin`
        - So it can be used for extending to get things moving quickly
        - And `is_loaded`, `is_available` are probably the most sensible defaults
    """

    __slots__ = ()
    sys_modules_targets: ClassVar[tuple[LiteralString_, ...]]
    implementation: ClassVar[KnownImpl]

    @property
    def plugin_name(self) -> LiteralString:
        return self.implementation.value  # type: ignore[no-any-return]

    def is_loaded(self) -> bool:  # pragma: no cover
        return all(sys.modules.get(target) for target in self.sys_modules_targets)

    def is_available(self) -> bool:  # pragma: no cover
        return all(find_spec(target) for target in self.sys_modules_targets)

    def native_classes(self) -> Iterator[type[DF | LF | S]]:  # pragma: no cover
        yield from self.native_dataframe_classes()
        yield from self.native_lazyframe_classes()
        yield from self.native_series_classes()

    def is_native(self, obj: Any) -> TypeIs[DF | LF | S]:  # pragma: no cover
        if tps := tuple(self.native_classes()):
            return isinstance(obj, tps)
        msg = f"{type(self).__name__!r} ({self.plugin_name!r}) has not defined any native classes"
        raise NotImplementedError(msg)

    def is_native_dataframe(self, obj: Any) -> TypeIs[DF]:  # pragma: no cover
        it = self.native_dataframe_classes()
        return bool(tps := tuple(it)) and isinstance(obj, tps)

    def is_native_lazyframe(self, obj: Any) -> TypeIs[LF]:  # pragma: no cover
        it = self.native_lazyframe_classes()
        return bool(tps := tuple(it)) and isinstance(obj, tps)

    def is_native_series(self, obj: Any) -> TypeIs[S]:  # pragma: no cover
        it = self.native_series_classes()
        return bool(tps := tuple(it)) and isinstance(obj, tps)


@functools.cache
def _entry_points() -> EntryPoints:
    from importlib.metadata import entry_points

    return entry_points(group="narwhals.plugins-plan")


@overload
def load_plugin(backend: Arrow, /) -> ArrowPlugin: ...
@overload
def load_plugin(backend: Polars, /) -> PolarsPlugin: ...
@overload
def load_plugin(backend: BackendTodo, /) -> Never: ...
@overload
def load_plugin(backend: NativeModuleType | Arrow | Polars, /) -> BuiltinAny: ...
@overload
def load_plugin(backend: PluginName, /) -> PluginAny: ...
def load_plugin(backend: IntoBackend[Backend] | PluginName, /) -> PluginAny | BuiltinAny:
    """Load the entry point to a backend.

    The returned object can be used to query availability.
    For built-ins, this is always safe and *does not* import the native package.
    """
    name = backend if isinstance(backend, str) else known_implementation(backend).value
    eps = _entry_points()
    plugin: PluginAny | BuiltinAny
    if found := eps.select(name=name):
        it = iter(found)
        plugin = next(it).load()
        if next(it, None) is None:
            return plugin
        msg = f"Multiple plugins found with the same name:\n{found!r}"  # pragma: no cover
        raise NotImplementedError(msg)
    raise _unsupported_error(backend, name, eps)


def _unsupported_error(backend: Any, name: str, eps: EntryPoints, /) -> Exception:
    if (impl := Implementation.from_backend(name)) is not Implementation.UNKNOWN:
        msg = f"{impl!r} is not yet supported in `narwhals._plan`, got: {backend!r}"
        return NotImplementedError(msg)
    msg = f"Unsupported `backend` value.\nExpected one of {sorted(eps.names)!r}, got: {backend!r}"
    return TypeError(msg)

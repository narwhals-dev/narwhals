"""Managing plugins."""

from __future__ import annotations

import functools
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,
    final,
    overload,
)

from narwhals._plan.plugins import _parse
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from importlib.metadata import EntryPoint, EntryPoints

    import polars as pl
    import pyarrow as pa
    from typing_extensions import Never, TypeAlias, TypeIs

    from narwhals._native import NativeDataFrame
    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.compliant import (
        CompliantDataFrame,
        CompliantLazyFrame,
        CompliantSeries,
        classes as cc,
        typing as ct,
    )
    from narwhals._plan.compliant.classes import C1, C2, CB, C
    from narwhals._plan.compliant.plugins import Builtin, Plugin
    from narwhals._plan.exceptions import unsupported_error
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.typing import (
        BackendTodo,
        BuiltinAny,
        IntoBackendExt,
        NativeModuleType,
        PluginAny,
        PluginName,
        VersionName,
    )
    from narwhals._typing import Arrow, BackendName, Polars
    from narwhals._typing_compat import assert_never


Incomplete: TypeAlias = Any

RequireMethod: TypeAlias = Literal["is_imported", "can_import"]
"""Raise if calling this method on `Plugin` returns False."""

_UNKNOWN: Final = Implementation.UNKNOWN
_GROUP: Final = "narwhals.plugins-plan"


# NOTE: https://github.com/python/mypy/issues/18786
_VERSION_NAME: Final[Mapping[Version, VersionName]] = {
    Version.MAIN: "MAIN",
    Version.V1: "V1",
    Version.V2: "V2",
}


# TODO @dangotbanned: (low-priority) Remove 3.10 guard after https://github.com/narwhals-dev/narwhals/issues/3204
# TODO @dangotbanned: (low-priority) Cover the duplicate name plugin case
@functools.cache
def _entry_points() -> EntryPoints:
    # NOTE: Wrappped with some one-time validation, so everything outside is simpler
    from importlib.metadata import entry_points

    if sys.version_info < (3, 10):
        msg = "Need `EntryPoints.{select,names}`, this can wait until 3.10 "
        raise NotImplementedError(msg)
    if (eps := entry_points(group=_GROUP)) and len(eps) == len(eps.names):
        return eps
    if not eps:  # pragma: no cover
        # If you're developing narwhals, this may have failed due to the `group` being renamed,
        # see `[project.entry-points.<group>]` in pyproject.toml
        call = f"{entry_points.__qualname__}(group={_GROUP!r})"
        msg = f"Expected to find built-in backends, but `{call}`\nreturned {eps!r}"
        raise NotImplementedError(msg)
    msg = f"Multiple plugins found with the same `name`:\n{eps!r}"  # pragma: no cover
    raise NotImplementedError(msg)


# TODO @dangotbanned: Fully remove all the overload experiments & `import_classes` tests
if TYPE_CHECKING:
    from typing_extensions import deprecated

    _R_co = TypeVar("_R_co", covariant=True)

    class _Plugin(Protocol[_R_co]):
        """Minimal interface for typing.

        Doesn't set a bound for `__narwhals_classes__`.
        """

        __slots__ = ()

        @property
        def name(self) -> PluginName: ...
        @property
        def __narwhals_classes__(self) -> _R_co: ...

    _PluginV1: TypeAlias = _Plugin["cc.HasV1[C1]"]
    _PluginV2: TypeAlias = _Plugin["cc.HasV2[C2]"]
    _PluginVAll: TypeAlias = _Plugin["cc.HasVAll[C1, C2]"]

    MAIN: TypeAlias = Literal[Version.MAIN]
    V1: TypeAlias = Literal[Version.V1]
    V2: TypeAlias = Literal[Version.V2]
    MYPY: Final = False
    if MYPY:
        # `Incomplete` avoids `mypy` from thinking overloads 1, 3, 5, 7 are bad
        _ImportClasses: TypeAlias = "_Plugin[C] | _Plugin[CB] | _PluginV1[C1] | _PluginV2[C2] | _PluginVAll[C1, C2] | _Plugin[Incomplete]"
    else:
        _ImportClasses: TypeAlias = "_Plugin[C] | _Plugin[CB] | _PluginV1[C1] | _PluginV2[C2] | _PluginVAll[C1, C2] | Builtin[CB, Any, Any, Any] | Plugin[C, Any, Any, Any]"

    # MAIN
    @overload  # 1
    def import_classes(plugin: _Plugin[CB], version: MAIN, /) -> CB: ...
    @overload  # 2
    def import_classes(plugin: _Plugin[C], version: MAIN, /) -> C: ...

    # V1
    @overload  # 3
    def import_classes(plugin: _PluginVAll[C1, C2], version: V1, /) -> C1: ...
    @overload  # 4
    def import_classes(plugin: _PluginV1[C1], version: V1, /) -> C1: ...

    # V2
    @overload  # 5
    def import_classes(plugin: _PluginVAll[C1, C2], version: V2, /) -> C2: ...
    @overload  # 6
    def import_classes(plugin: _PluginV2[C2], version: V2, /) -> C2: ...

    # OPAQUE
    # NOTE: `_Plugin[C] | _Plugin[CB]` avoids `PluginAny` from reporting as a builtin
    # when we have `(PluginAny | BuiltinAny, Version)`
    @overload  # 7
    def import_classes(
        plugin: _Plugin[C]
        | _Plugin[CB]
        | _Plugin[cc.EagerImplC]
        | _PluginV1[C1]
        | _PluginV2[C2]
        | _PluginVAll[C1, C2]
        | _PluginVAll[cc.CB1, cc.CB2],
        version: Version,
        /,
    ) -> C | CB | C1 | C2 | cc.CB1 | cc.CB2 | cc.EagerImplC: ...

    if MYPY:
        # avoids `mypy` deciding that every type parameter is `Never`
        @overload  # 8
        def import_classes(
            plugin: _Plugin[Incomplete], version: Version, /
        ) -> Incomplete: ...

    @deprecated("overloads are too LSP heavy, need something simpler")
    def import_classes(
        plugin: _ImportClasses[C, CB, C1, C2], version: Version, /
    ) -> Incomplete:
        """Import the accessor to the compliant classes compatible with `version`."""
        classes = plugin.__narwhals_classes__
        if version is Version.MAIN:
            return classes
        if version is Version.V1:
            if cc.can_v1(classes):
                return classes.v1
            raise unsupported_error(plugin.name, "v1")
        if version is Version.V2:
            if cc.can_v2(classes):
                return classes.v2
            raise unsupported_error(plugin.name, "v2")
        assert_never(version)


# TODO @dangotbanned: Integrate `_parsed`, `_registry`
# TODO @dangotbanned: Add somewhere for unreachable plugins to live
@final
class PluginManager:
    """Singleton plugin manager.

    ## Notes
    - if there is state, how can we avoid knowledge of that leaking everywhere?
        - it's okay for state to exist
        - but shouldn't be something the caller has to deal with
            - parsing/error handling stays within it
            - maybe allow providing an error message on fail
    - 3 groups of plugins
        - `is_imported()`
        - `(not is_imported()) and can_import()`
            - Allow access to the plugin when explicitly asked (`to_*`, `backend=*`)
            - During inference, check here when all of `is_imported` is exhausted
            - Promote to `is_imported` when found
      - `not can_import()`
            - Once here, the plugin is unreachable
            - Stop checking
    """

    __slots__ = ("_discovered", "_loaded", "_parsed", "_registry")
    __instance: ClassVar[Any | None] = None

    _discovered: dict[PluginName, EntryPoint]
    _loaded: dict[PluginName, PluginAny | BuiltinAny]

    _parsed: dict[PluginName, _parse.PluginIR]
    """Details on what each plugin supports."""

    _registry: dict[PluginName, _parse.RegEntry]
    """A rewrapped `Plugin`, with error handling on unsupported features."""

    def __new__(cls) -> PluginManager:
        if not isinstance(cls.__instance, PluginManager):
            self = object.__new__(PluginManager)
            # NOTE: Need to lie about `LiteralString` because `str` leaks to all other usage
            _eps = {ep.name: ep for ep in _entry_points()}
            self._discovered = cast("dict[PluginName, EntryPoint]", _eps)  # type: ignore[redundant-cast]
            self._loaded = {}
            self._parsed = {}
            self._registry = {}
            cls.__instance = self
        return cls.__instance

    def __repr__(self) -> str:
        n_discovered = len(self._discovered)
        n_loaded = len(self._loaded)
        indent = " " * 4
        indent_2 = indent * 2
        join = f"\n{indent_2}".join
        s = ""
        if n_loaded:
            s_plugins = join(map(repr, self._loaded.values()))
            s += f"\n{indent}loaded\n{indent_2}{s_plugins}"
        if n_discovered:
            s_eps = join(f"EntryPoint[{name}]" for name in self._discovered)
            s += f"\n{indent}discovered\n{indent_2}{s_eps}"
        return f"{type(self).__name__}[{n_discovered + n_loaded}]{s}"

    def _plugin_load(self, name: PluginName, entry_point: EntryPoint, /) -> PluginAny:
        # NOTE: Keeps `_plugin` and `_iter_plugins` in sync
        plugin: PluginAny
        self._loaded[name] = plugin = entry_point.load()
        return plugin

    def _plugin_parse(self, name: PluginName, /) -> _parse.PluginIR:
        """Discover features supported by a plugin, without invoking native imports."""
        if parsed := self._parsed.get(name):
            return parsed
        self._parsed[name] = ir = _parse.PluginIR.from_plugin(self._plugin(name))
        return ir

    def _plugin_entry(self, name: PluginName, /) -> _parse.RegEntry:
        """Lower a plugin into a proxy, providing error wrapping for missing features."""
        registry = self._registry
        if entry := registry.get(name):
            return entry
        registry[name] = entry = self._plugin_parse(name).to_registry_entry()
        return entry

    def _plugin(self, name: PluginName, /) -> PluginAny:
        if loaded := self._loaded.get(name):
            return loaded
        if entry_point := self._discovered.pop(name, None):
            return self._plugin_load(name, entry_point)
        raise _unsupported_error(name, name)

    def _iter_plugins(self) -> Iterator[PluginAny | BuiltinAny]:
        yield from self._loaded.values()
        while self._discovered:  # pragma: no cover
            name, ep = self._discovered.popitem()
            yield self._plugin_load(name, ep)

    def _get_class(
        self, name: cc.PropertyName, backend: IntoBackendExt, version: Version, /
    ) -> type[Any]:
        plugin_name = _backend_to_plugin_name(backend)
        classes = self._plugin(plugin_name).__narwhals_classes__
        return self._plugin_entry(plugin_name)[_VERSION_NAME[version]][name](classes)

    @overload
    def plugin(
        self, backend: Arrow, /, require: RequireMethod | None = None
    ) -> ArrowPlugin: ...
    @overload
    def plugin(
        self, backend: Polars, /, require: RequireMethod | None = None
    ) -> PolarsPlugin: ...
    @overload
    def plugin(
        self, backend: BackendTodo, /, require: RequireMethod | None = None
    ) -> Never: ...
    @overload
    def plugin(
        self,
        backend: NativeModuleType | Arrow | Polars,
        /,
        require: RequireMethod | None = None,
    ) -> BuiltinAny: ...
    @overload
    def plugin(
        self, backend: PluginName, /, require: RequireMethod | None = None
    ) -> PluginAny: ...
    @overload
    def plugin(
        self, backend: IntoBackendExt, /, require: RequireMethod | None = None
    ) -> PluginAny | BuiltinAny: ...
    def plugin(
        self, backend: IntoBackendExt, /, require: RequireMethod | None = None
    ) -> PluginAny | BuiltinAny:
        """Return the plugin matching `backend`, raising if `require` returns False."""
        plugin = self._plugin(_backend_to_plugin_name(backend))
        if not require:
            return plugin
        if (
            plugin.is_imported() if require == "is_imported" else plugin.can_import()
        ):  # pragma: no cover
            return plugin
        raise _unavailable_error(plugin, require)  # pragma: no cover

    # NOTE: These overloads are *intentionally* less-precise than they could be
    # Some early experiments handled unions & version matching (successfuly),
    # but came at a very high cost to LSP performance.
    # **Before adding more complexity here again - consider operating on the plugin directly.**
    @overload
    def dataframe(
        self, backend: Polars, /, version: Version
    ) -> type[CompliantDataFrame[pl.DataFrame, pl.Series]]: ...
    @overload
    def dataframe(
        self, backend: Arrow, /, version: Version
    ) -> type[CompliantDataFrame[pa.Table, pa.ChunkedArray[Any]]]: ...
    @overload
    def dataframe(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.DataFrameAny]: ...
    def dataframe(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.DataFrameAny]:
        return self._get_class("_dataframe", backend, version)

    @overload
    def series(
        self, backend: Polars, /, version: Version
    ) -> type[CompliantSeries[pl.Series]]: ...
    @overload
    def series(
        self, backend: Arrow, /, version: Version
    ) -> type[CompliantSeries[pa.ChunkedArray[Any]]]: ...
    @overload
    def series(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.SeriesAny]: ...
    def series(self, backend: IntoBackendExt, /, version: Version) -> type[ct.SeriesAny]:
        return self._get_class("_series", backend, version)

    @overload
    def lazyframe(
        self, backend: Polars, /, version: Version
    ) -> type[CompliantLazyFrame[pl.LazyFrame]]: ...
    @overload
    def lazyframe(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.LazyFrameAny]: ...
    def lazyframe(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.LazyFrameAny]:
        return self._get_class("_lazyframe", backend, version)

    @overload
    def evaluator(
        self, backend: Polars, /, version: Version
    ) -> type[ct.PlanEvaluator[pl.LazyFrame]]: ...
    @overload
    def evaluator(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.PlanEvaluatorAny]: ...
    def evaluator(
        self, backend: IntoBackendExt, /, version: Version
    ) -> type[ct.PlanEvaluatorAny]:
        return self._get_class("_evaluator", backend, version)

    # TODO @dangotbanned: Use in `@singledispatch` (see stash `expr-ir-from-native-series-dispatch`)
    def native_dataframe_classes(self) -> Iterator[type[NativeDataFrame]]:
        for plugin in self._iter_plugins():
            if plugin.is_imported() and self._plugin_parse(plugin.name).main.dataframe:
                yield from plugin.native_dataframe_classes()
            else:  # pragma: no cover
                ...

    def is_native_dataframe(self, native: Any) -> TypeIs[NativeDataFrame]:
        return bool(types := tuple(self.native_dataframe_classes())) and isinstance(
            native, types
        )

    def import_modules(self, backend: IntoBackendExt, /) -> None:
        """Import the requirements for `backend`."""
        plugin = self.plugin(backend, require="can_import")
        if not plugin.is_imported():
            for module in plugin.requirements:
                import_module(module)

    def importable(self) -> Iterator[BackendName | PluginName]:
        """Yield the names of all importable plugins."""
        for plugin in self._iter_plugins():
            if plugin.can_import():
                yield plugin.name
            else:  # pragma: no cover
                continue

    def imported(self) -> Iterator[BackendName | PluginName]:
        """Yield the names of all imported plugins."""
        for plugin in self._iter_plugins():
            if plugin.is_imported():
                yield plugin.name

    def known(self) -> Iterator[BackendName | PluginName | str]:
        """Yield the names of all known plugins."""
        yield from (ep.name for ep in _entry_points())

    def show_versions(self) -> None:
        raise NotImplementedError


def _backend_to_plugin_name(backend: IntoBackendExt, /) -> BackendName | PluginName:
    if isinstance(backend, str):
        return backend
    if backend is _UNKNOWN or (impl := Implementation.from_backend(backend)) is _UNKNOWN:
        msg = f"{_UNKNOWN!r} is not supported in this context, got: {backend!r}"
        raise NotImplementedError(msg)
    name: BackendName = impl.value
    return name


def _unsupported_error(backend: Any, name: str, /) -> Exception:
    if (impl := Implementation.from_backend(name)) is not _UNKNOWN:
        msg = f"{impl!r} is not yet supported in `narwhals._plan`, got: {backend!r}"
        return NotImplementedError(msg)
    msg = f"Unsupported `backend` value.\nExpected one of {sorted(_entry_points().names)!r}, got: {backend!r}"
    return TypeError(msg)


def _unavailable_error(
    plugin: PluginAny, require: RequireMethod = "can_import"
) -> Exception:  # pragma: no cover
    msg = f"Plugin {plugin.name!r} was found but"
    if require == "is_imported" and plugin.can_import():
        missing = [name for name in plugin.requirements if sys.modules.get(name) is None]
        reason = "the following available modules have not yet been imported"
    else:
        missing = [name for name in plugin.requirements if find_spec(name) is None]
        reason = "could not import the following required modules"
    msg = f"{msg} {reason}: {missing!r}"
    return ModuleNotFoundError(msg)

"""Runtime discovery of, and dispatch to, Narwhals plugin backends.

A plugin backend registers an [entry point](https://packaging.python.org/en/latest/specifications/entry-points/)
in the `narwhals.plugins` group. Plugins are discovered at runtime, so their names
cannot be enumerated in the `Literal` unions that describe built-in backends.

`PluginName` bridges that gap: a plugin's entry point name, wrapped as
`PluginName("my-plugin")`, is accepted wherever a `backend` is expected.

The contract for plugin authors is that the wrapped string **must** name an
installed plugin's entry point in the `narwhals.plugins` group.
"""

from __future__ import annotations

import sys
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

from narwhals._compliant import CompliantNamespace
from narwhals._typing import PluginName
from narwhals._typing_compat import TypeVar
from narwhals.exceptions import PluginError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from importlib.metadata import EntryPoints
    from typing import TypeAlias

    from typing_extensions import LiteralString

    from narwhals._compliant.typing import (
        CompliantDataFrameAny,
        CompliantFrameAny,
        CompliantLazyFrameAny,
        CompliantSeriesAny,
    )
    from narwhals._typing import Backend, IntoBackend
    from narwhals.utils import Version

    IOMethodName: TypeAlias = Literal[
        "read_csv", "read_parquet", "scan_csv", "scan_parquet"
    ]
    """Name of a Narwhals IO function, dispatched to a same-named namespace method."""


__all__ = ["Plugin", "PluginName", "from_native"]

CompliantAny: TypeAlias = (
    "CompliantDataFrameAny | CompliantLazyFrameAny | CompliantSeriesAny"
)
"""A statically-unknown, Compliant object originating from a plugin."""

FrameT = TypeVar(
    "FrameT",
    bound="CompliantFrameAny",
    default="CompliantDataFrameAny | CompliantLazyFrameAny",
)
FromNativeR_co = TypeVar(
    "FromNativeR_co", bound=CompliantAny, covariant=True, default=CompliantAny
)


@cache
def _discover_entrypoints() -> EntryPoints:
    from importlib.metadata import entry_points as eps

    group = "narwhals.plugins"
    return eps(group=group)


def _plugin_names() -> tuple[str, ...]:
    """Entry point names of all installed plugins."""
    return tuple(entry_point.name for entry_point in _discover_entrypoints())


def _find_plugin(backend_name: str, /) -> Plugin | None:
    """Return the first installed plugin matching `backend_name`.

    `backend_name` is matched against both the entry point name and its module,
    e.g. both `"my-plugin"` and `"my_plugin"` for a plugin registered as:

        [project.entry-points.'narwhals.plugins']
        my-plugin = 'my_plugin'

    Note:
        The parameter is a plain `str`, not a `PluginName`: the module spelling is a
        valid input and is *not* an entry point name.
    """
    for entry_point in _discover_entrypoints():
        if backend_name in {entry_point.name, entry_point.module}:
            plugin: Plugin = entry_point.load()
            return plugin
    return None


def _backend_namespace(backend: IntoBackend[Backend | PluginName], /) -> Plugin:
    """Resolve a backend which is not a Narwhals `Implementation` to a plugin.

    The plugin is expected to implement the `Plugin` protocol, in particular the
    `__narwhals_namespace__` function returning a compliant namespace.
    """
    if isinstance(backend, ModuleType):
        # NOTE: A user-provided module is only *claimed* to be a `Plugin`; the runtime
        # guard is `_plugin_namespace`, which raises `PluginError` if it is not one.
        return cast("Plugin", backend)
    if isinstance(backend, str) and (plugin := _find_plugin(backend)) is not None:
        return plugin
    installed = ", ".join(_plugin_names()) or "<none>"
    msg = (
        f"Unsupported backend: {backend!r}.\n\n"
        "Expected one of Narwhals' built-in backends (e.g. 'pandas', 'polars', "
        "'pyarrow'), a native namespace module, or the name of an installed "
        f"Narwhals plugin (installed plugins: {installed})."
    )
    raise ValueError(msg)


def _plugin_namespace(plugin: Plugin, /, *, version: Version) -> PluginNamespace:
    """Get `plugin`'s compliant namespace, raising if `__narwhals_namespace__` is missing."""
    name = "__narwhals_namespace__"
    if (hook := getattr(plugin, name, None)) is None:
        msg = f"Plugin backend {plugin.__name__!r} is expected to implement `{name}` function."
        raise PluginError(msg)
    namespace: PluginNamespace = hook(version=version)
    return namespace


@overload
def plugin_io_method(
    backend: IntoBackend[Backend | PluginName],
    method_name: Literal["read_csv", "read_parquet"],
    /,
    *,
    version: Version,
) -> Callable[..., CompliantDataFrameAny]: ...


@overload
def plugin_io_method(
    backend: IntoBackend[Backend | PluginName],
    method_name: Literal["scan_csv", "scan_parquet"],
    /,
    *,
    version: Version,
) -> Callable[..., CompliantFrameAny]: ...


def plugin_io_method(
    backend: IntoBackend[Backend | PluginName],
    method_name: IOMethodName,
    /,
    *,
    version: Version,
) -> Callable[..., CompliantFrameAny]:
    """Resolve `backend` to the `method_name` method of a plugin's compliant namespace.

    IO functions share a single dispatch mechanism with built-in backends: they call
    same-named methods on the compliant namespace (see the "IO functions" section of
    the [extension docs](../extending.md/#io-functions-the-namespace-contract)).

    Note:
        `PluginNamespace` deliberately does not declare the IO methods: they are an
        optional subset of a plugin (e.g. a lazy-only plugin implements `scan_*` only).
    """
    from inspect import getattr_static

    from narwhals._utils import not_implemented

    plugin = _backend_namespace(backend)
    namespace = _plugin_namespace(plugin, version=version)
    method = getattr_static(namespace, method_name, None)
    if method is None or isinstance(method, not_implemented):
        msg = (
            f"Plugin backend {plugin.__name__!r} is expected to implement "
            f"`{method_name}` on its compliant namespace to support `narwhals.{method_name}`."
        )
        raise PluginError(msg)
    bound_method: Callable[..., CompliantFrameAny] = getattr(namespace, method_name)
    return bound_method


class PluginNamespace(CompliantNamespace[FrameT, Any], Protocol[FrameT, FromNativeR_co]):
    """A `CompliantNamespace` which can also wrap native objects via `from_native`."""

    def from_native(self, data: Any, /) -> FromNativeR_co:
        """Wrap a native object into a compliant DataFrame, LazyFrame, or Series."""
        ...


class Plugin(Protocol[FrameT, FromNativeR_co]):
    """Top-level interface a plugin module is expected to implement.

    A plugin is a module registered in the `narwhals.plugins`
    [entry point](https://packaging.python.org/en/latest/specifications/entry-points/)
    group:

    ```toml
    [project.entry-points.'narwhals.plugins']
    narwhals-grizzlies = 'narwhals_grizzlies'
    ```

    Narwhals discovers installed plugins at runtime and uses this interface to
    recognise their native objects (`NATIVE_PACKAGE`, `is_native`) and to obtain
    a compliant namespace (`__narwhals_namespace__`), through which all further
    dispatch happens.

    See [extensions and plugins](../extending.md) for a complete walk-through.
    """

    @property
    def __name__(self) -> str:
        """Name of the plugin module, used to identify the backend in error messages.

        Automatically provided: a plugin *is* a module, and every module has a `__name__`.
        """
        ...

    @property
    def NATIVE_PACKAGE(self) -> LiteralString:  # noqa: N802
        """Name of the package providing the plugin's native objects, e.g. `"grizzlies"`.

        Used as a cheap pre-check when converting native objects: the plugin is only
        consulted if this package is already imported and the inspected object's class
        might originate from it.
        """
        ...

    def __narwhals_namespace__(
        self, version: Version
    ) -> PluginNamespace[FrameT, FromNativeR_co]:
        """Return a compliant namespace for the given Narwhals API version.

        The returned namespace is the plugin's dispatch hub: its `from_native` method
        wraps native objects, IO functions call its `scan_*`/`read_*` methods, and
        eager constructors use its `_dataframe`/`_series` classes (see the
        [`backend=...` section](../extending.md/#supporting-backend-in-narwhals-functions)
        of the extension docs).
        """
        ...

    def is_native(self, native_object: object, /) -> bool:
        """Return whether `native_object` is a native object of the plugin's library."""
        ...


@cache
def _might_be(cls: type, type_: str) -> bool:  # pragma: no cover
    try:
        return any(type_ in o.__module__.split(".") for o in cls.mro())
    except TypeError:
        return False


def _is_native_plugin(native_object: Any, plugin: Plugin) -> bool:
    pkg = plugin.NATIVE_PACKAGE
    return (
        sys.modules.get(pkg) is not None
        and _might_be(type(native_object), pkg)  # type: ignore[arg-type]
        and plugin.is_native(native_object)
    )


def _iter_from_native(native_object: Any, version: Version) -> Iterator[CompliantAny]:
    for entry_point in _discover_entrypoints():
        plugin: Plugin = entry_point.load()
        if _is_native_plugin(native_object, plugin):
            compliant_namespace = plugin.__narwhals_namespace__(version=version)
            yield compliant_namespace.from_native(native_object)


def from_native(native_object: Any, version: Version) -> CompliantAny | None:
    """Attempt to convert `native_object` to a Compliant object, using any available plugin(s).

    Arguments:
        native_object: Raw object from user.
        version: Narwhals API version.

    Returns:
        If the following conditions are met

            - at least 1 plugin is installed
            - at least 1 installed plugin supports `type(native_object)`

            Then for the **first matching plugin**, the result of the call below.

            This *should* be an object accepted by a Narwhals Dataframe, Lazyframe, or Series:

                plugin: Plugin
                plugin.__narwhals_namespace__(version).from_native(native_object)

            In all other cases, `None` is returned instead.
    """
    return next(_iter_from_native(native_object, version), None)


def is_native_dataframe(native_object: Any) -> bool:
    """Check whether an installed plugin converts `native_object` to an eager DataFrame."""
    from narwhals._utils import Version, is_compliant_dataframe

    return is_compliant_dataframe(from_native(native_object, Version.MAIN))


def is_native_lazyframe(native_object: Any) -> bool:
    """Check whether an installed plugin converts `native_object` to a LazyFrame."""
    from narwhals._utils import Version, is_compliant_lazyframe

    return is_compliant_lazyframe(from_native(native_object, Version.MAIN))


def is_native_series(native_object: Any) -> bool:
    """Check whether an installed plugin converts `native_object` to a Series."""
    from narwhals._utils import Version, is_compliant_series

    return is_compliant_series(from_native(native_object, Version.MAIN))


def _show_suggestions(native_object_type: type) -> str | None:
    if _might_be(native_object_type, "daft"):  # pragma: no cover
        return (
            "Hint: it looks like you passed a `daft.DataFrame` but don't have `narwhals-daft` installed.\n"
            "Please refer to https://github.com/narwhals-dev/narwhals-daft for installation instructions."
        )
    return None

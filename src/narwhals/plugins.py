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
from typing import TYPE_CHECKING, Any, Protocol, cast

from narwhals._compliant import CompliantNamespace, EagerNamespace
from narwhals._typing import PluginName
from narwhals._typing_compat import TypeVar
from narwhals.exceptions import PluginError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.metadata import EntryPoints
    from typing import TypeAlias

    from typing_extensions import LiteralString

    from narwhals._compliant.typing import (
        CompliantDataFrameAny,
        CompliantFrameAny,
        CompliantLazyFrameAny,
        CompliantNamespaceAny,
        CompliantSeriesAny,
    )
    from narwhals._typing import IntoBackend, IOMethodName
    from narwhals.utils import Version


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


# TODO(Unassigned): https://github.com/narwhals-dev/narwhals/issues/4026
@cache
def _find_plugin(backend_name: str, /) -> Plugin | None:
    """Return the first installed plugin whose entry point name or module is `backend_name`.

    For an entry point `my-plugin = 'my_plugin'`, both `"my-plugin"` and `"my_plugin"`
    match, which is why `backend_name` is a plain `str` rather than a `PluginName`.
    """
    for entry_point in _discover_entrypoints():
        if backend_name in {entry_point.name, entry_point.module}:
            plugin: Plugin = entry_point.load()
            return plugin
    return None


def _plugin_display_name(backend: IntoBackend[PluginName], /) -> str:
    return backend.__name__ if isinstance(backend, ModuleType) else backend


def _resolve_plugin(backend: IntoBackend[PluginName], /) -> Plugin:
    if isinstance(backend, ModuleType):
        # NOTE: Unverified until `_plugin_namespace` looks up `__narwhals_namespace__`.
        return cast("Plugin", backend)
    if (plugin := _find_plugin(backend)) is not None:
        return plugin
    installed = ", ".join(ep.name for ep in _discover_entrypoints()) or "<none>"
    msg = (
        f"Unsupported backend: {backend!r}.\n\n"
        "Expected one of Narwhals' built-in backends (e.g. 'pandas', 'polars', "
        "'pyarrow'), a native namespace module, or the name of an installed "
        f"Narwhals plugin (installed plugins: {installed})."
    )
    raise ValueError(msg)


# NOTE: A `dict`, not `functools.cache`: mypy never treats a `Protocol` (`Plugin`) as `Hashable`.
_PLUGIN_NAMESPACES: dict[tuple[Plugin, Version], PluginNamespace] = {}


def _plugin_namespace(plugin: Plugin, /, *, version: Version) -> PluginNamespace:
    key = (plugin, version)
    if (namespace := _PLUGIN_NAMESPACES.get(key)) is None:
        name = "__narwhals_namespace__"
        if (hook := getattr(plugin, name, None)) is None:
            msg = f"Plugin backend {plugin.__name__!r} is expected to implement `{name}` function."
            raise PluginError(msg)
        # NOTE: `setdefault`, so that concurrent first calls still share one namespace.
        namespace = _PLUGIN_NAMESPACES.setdefault(key, hook(version=version))
    return namespace


# TODO(Unassigned): https://github.com/narwhals-dev/narwhals/issues/4025
def _ensure_io_method(
    namespace: CompliantNamespaceAny, method_name: IOMethodName, /, *, plugin_name: str
) -> None:
    """Raise unless a plugin's compliant namespace implements `method_name`.

    See the [IO functions](../extending.md/#io-functions-the-namespace-contract) contract.

    Note:
        `PluginNamespace` deliberately does not declare the IO methods: they are an
        optional subset of a plugin (e.g. a lazy-only plugin implements `scan_*` only).
        A `not_implemented` placeholder and an inherited protocol stub count as missing.
    """
    from inspect import getattr_static

    from narwhals._utils import not_implemented

    # NOTE: `EagerNamespace.scan_*` are real defaults (falling back to `read_*`), not stubs.
    protocol = CompliantNamespace if method_name.startswith("scan_") else EagerNamespace
    method = getattr_static(namespace, method_name, None)
    if (
        method is None
        or isinstance(method, not_implemented)
        or method is getattr_static(protocol, method_name)
    ):
        msg = (
            f"Plugin backend {plugin_name!r} is expected to implement "
            f"`{method_name}` on its compliant namespace to support `narwhals.{method_name}`."
        )
        raise PluginError(msg)


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
        """Name of the plugin module, used in error messages. Every module has one."""
        ...

    @property
    def NATIVE_PACKAGE(self) -> LiteralString:  # noqa: N802
        """Name of the package providing the plugin's native objects, e.g. `"grizzlies"`.

        Narwhals only consults the plugin about an object if this package is already
        imported and the object's class might come from it.
        """
        ...

    def __narwhals_namespace__(
        self, version: Version
    ) -> PluginNamespace[FrameT, FromNativeR_co]:
        """Return a compliant namespace for the given Narwhals API version.

        Narwhals wraps native objects with its `from_native` method, dispatches IO
        functions to its `scan_*`/`read_*` methods, and builds eager constructors from
        its `_dataframe`/`_series` classes (see
        [`backend=...`](../extending.md/#supporting-backend-in-narwhals-functions)).

        Important:
            Narwhals caches the returned namespace per version and shares it across calls,
            so it must be safe to reuse. The hook itself may still run more than once,
            e.g. on concurrent first use.
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
            yield _plugin_namespace(plugin, version=version).from_native(native_object)


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
    if _might_be(native_object_type, "datafusion"):  # pragma: no cover
        return (
            "Hint: it looks like you passed a `datafusion.DataFrame` but don't have `narwhals-datafusion` installed.\n"
            "Please refer to https://github.com/s5dsn-eqee/narwhals-datafusion for installation instructions."
        )
    return None

"""Plugins API."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from narwhals._plan import _plugins
from narwhals._plan.compliant.plugins import Builtin, Plugin

if TYPE_CHECKING:
    from typing_extensions import Never

    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.typing import (
        BackendTodo,
        BuiltinAny,
        IntoBackendExt,
        NativeModuleType,
        PluginAny,
        PluginName,
    )
    from narwhals._typing import Arrow, Polars

__all__ = ("Builtin", "Plugin", "load_plugin")


# TODO @dangotbanned: Think about renaming this after moving tests/docs to using it from here
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
def load_plugin(backend: IntoBackendExt, /) -> PluginAny:
    """Load the entry point to a backend.

    The returned object can be used to query availability.
    For built-ins, this is always safe and *does not* import the native package.
    """
    return _plugins.PluginManager().get(backend)

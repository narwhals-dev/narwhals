from __future__ import annotations

import sys
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from importlib.metadata import EntryPoints

    from typing_extensions import LiteralString

    from narwhals._compliant.typing import CompliantNamespaceAny
    from narwhals.utils import Version


@cache
def discover_entrypoints() -> EntryPoints:
    from importlib.metadata import entry_points as eps

    group = "narwhals.plugins"
    if sys.version_info < (3, 10):
        return cast("EntryPoints", eps().get(group, ()))
    return eps(group=group)


class Plugin(Protocol):
    NATIVE_PACKAGE: LiteralString

    def __narwhals_namespace__(self, version: Version) -> CompliantNamespaceAny: ...
    def is_native(self, native_object: object, /) -> bool: ...


@cache
def _might_be(cls: type, type_: str) -> bool:
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

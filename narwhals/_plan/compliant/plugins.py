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
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from narwhals._plan.common import hasattrs_static
from narwhals._plan.compliant.classes import CB, C, HasClasses
from narwhals._plan.compliant.typing import (
    Native as LF,
    NativeDataFrameT as DF,
    NativeSeriesT as S,
)
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.typing import KnownImpl, PluginName
    from narwhals._typing import BackendName

    MYPY: Final = False
    if MYPY:
        # NOTE: Use this to avoid mypy's PEP 675 hack from blocking the feature
        # https://github.com/python/mypy/issues/12554
        LiteralString_: TypeAlias = Any
    else:
        from typing_extensions import LiteralString as LiteralString_


__all__ = ("Builtin", "Implementation", "Plugin")


class Plugin(HasClasses[C], Protocol[C, DF, LF, S]):
    """An entrypoint for a backend that implements compliant-level operations.

    `[C, DF, LF, S]`.
    """

    __slots__ = ()
    requirements: ClassVar[tuple[LiteralString_, ...]]
    """One or more native package/module names which must be available to use this plugin."""

    # TODO @dangotbanned: (low-priority) Populate using `EntryPoint.name`?
    @property
    def name(self) -> PluginName: ...

    def __hash__(self) -> int:
        # Each plugin must have a unique name
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        # Loose guard to avoid `self == self.name`
        names = "requirements", "native_classes"
        return hasattrs_static(other, *names) and hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"Plugin[{self.name}]"

    def is_imported(self) -> bool:
        """Return True if all required dependencies *have already been* imported.

        ## Examples
        We use this to detect which backend to use when provided a native object:
        >>> # User
        >>> import polars as pl
        >>> native = pl.DataFrame()

        For `native` to *possibly* be a `pl.DataFrame`, that import had to have happened:
        >>> # Narwhals
        >>> from narwhals._plan.plugins import load_plugin
        >>> plugin = load_plugin("polars")
        >>> plugin
        Plugin[polars]
        >>> plugin.is_imported()
        True

        Now we can freely import `polars`, without requiring the dependency:
        >>> import polars as pl
        >>> isinstance(native, pl.DataFrame)
        True

        This check does not require reloading the plugin to update:
        >>> _ = sys.modules.pop("pyarrow", None)
        >>> plugin = load_plugin("pyarrow")
        >>> plugin
        Plugin[pyarrow]
        >>> plugin.is_imported()
        False
        >>> import pyarrow
        >>> plugin.is_imported()
        True
        """
        # NOTE: `sys.modules` may be populated by other tests in any order.
        # Removing `"pyarrow"` ensures the result of this one isn't contaminated
        ...

    def can_import(self) -> bool:
        """Return True if we *can* import all required dependencies.

        Important:
            Prefer `is_imported` if you only need the import for a runtime type check.

        ## Examples
        We use this for operations that convert between backends:
        >>> from narwhals._plan.plugins import load_plugin
        >>> import polars as pl
        >>> native = pl.Series([1, 2, 3])
        >>> load_plugin("pyarrow").can_import()
        True

        Now we can be sure this won't raise an `ImportError`:
        >>> native.to_arrow()
        <pyarrow.lib.Int64Array ...
        [
          1,
          2,
          3
        ]
        """
        ...

    def is_native(self, obj: Any, /) -> TypeIs[DF | LF | S]: ...
    def is_native_dataframe(self, obj: Any, /) -> TypeIs[DF]: ...
    def is_native_lazyframe(self, obj: Any, /) -> TypeIs[LF]: ...
    def is_native_series(self, obj: Any, /) -> TypeIs[S]: ...

    def native_dataframe_classes(self) -> Iterator[type[DF]]: ...
    def native_lazyframe_classes(self) -> Iterator[type[LF]]: ...
    def native_series_classes(self) -> Iterator[type[S]]: ...
    def native_classes(self) -> Iterator[type[DF | LF | S]]: ...


# TODO @dangotbanned: (low-priority) Preserve the exact `implementation` for each backend (bad overloads)
class Builtin(Plugin[CB, DF, LF, S], Protocol[CB, DF, LF, S]):
    """Backends defined inside of narwhals are plugins too.

    `[CB, DF, LF, S]`.

    ## Notes
    - Might want to provide *parts* of this in a sub-protocol for `Plugin`
    - So it can be used for extending to get things moving quickly
    - And `is_imported`, `can_import` are probably the most sensible defaults
    """

    __slots__ = ()
    implementation: ClassVar[KnownImpl]

    @property
    def name(self) -> BackendName:
        name: BackendName = self.implementation.value
        return name

    def is_imported(self) -> bool:
        return all(sys.modules.get(target) for target in self.requirements)

    def can_import(self) -> bool:
        return _can_import(self)

    def native_classes(self) -> Iterator[type[DF | LF | S]]:
        yield from self.native_dataframe_classes()
        yield from self.native_lazyframe_classes()
        yield from self.native_series_classes()

    def is_native(self, obj: Any) -> TypeIs[DF | LF | S]:  # pragma: no cover
        if tps := tuple(self.native_classes()):
            return isinstance(obj, tps)
        msg = (
            f"{type(self).__name__!r} ({self.name!r}) has not defined any native classes"
        )
        raise NotImplementedError(msg)

    def is_native_dataframe(self, obj: Any) -> TypeIs[DF]:
        it = self.native_dataframe_classes()
        return bool(tps := tuple(it)) and isinstance(obj, tps)

    def is_native_lazyframe(self, obj: Any) -> TypeIs[LF]:
        it = self.native_lazyframe_classes()
        return bool(tps := tuple(it)) and isinstance(obj, tps)

    def is_native_series(self, obj: Any) -> TypeIs[S]:
        it = self.native_series_classes()
        return bool(tps := tuple(it)) and isinstance(obj, tps)


@functools.cache
def _can_import(plugin: Plugin[Any, Any, Any, Any], /) -> bool:
    """Cached `Plugin.can_import`, bounded by the total number of plugins."""
    return all(find_spec(target) for target in plugin.requirements)

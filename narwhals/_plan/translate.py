"""Lazy, plugin-friendly, single-dispatch, doubly-registered and triply-magical `from_native_*` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

from narwhals._plan.plugins._manager import PluginManager
from narwhals._utils import Version, qualified_type_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from typing_extensions import ParamSpec, TypeAlias, TypeIs

    from narwhals._native import NativeDataFrame, NativeLazyFrame, NativeSeries
    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.compliant.typing import (
        Native as LF,
        NativeDataFrameT as DF,
        NativeSeriesT as S,
    )
    from narwhals._plan.plugins._typing import (
        SingleDispatchCallable,
        from_native_dispatch as singledispatch,
    )
    from narwhals._plan.typing import PluginAny, PluginName

    P = ParamSpec("P")

else:  # pragma: no cover
    import sys
    from functools import singledispatch

    # TODO @dangotbanned: Remove this hack once 3.9 is dropped
    # fmt: off
    if sys.version_info >= (3, 10):
        from collections.abc import Callable
        from typing import ParamSpec
        P = ParamSpec("P")
    else:
        _T1 = TypeVar("_T1")
        _T2 = TypeVar("_T2")
        class Callable(Generic[_T1, _T2]): ...
        class ParamSpec:
            args = Any
            kwargs = Any
        P = ParamSpec()
# fmt: on

__all__ = ("from_native_dataframe", "from_native_lazyframe", "from_native_series")

Incomplete: TypeAlias = Any
R_co = TypeVar(
    "R_co", bound="ct.DataFrameAny | ct.LazyFrameAny | ct.SeriesAny", covariant=True
)
TranslateName: TypeAlias = Literal["dataframe", "lazyframe", "series"]


# TODO @dangotbanned: Docs need a lot of love!
# - Frankensteined x2 w/ previous one from `translate.py` + fleeting `PluginManager._find_from_native`
class _FromNative(Generic[P, R_co]):
    """Self-registration dispatcher.

    ## Super high-level
    - `@singledispatch` starts with no registered implementations
        - We search here through a metadata registry
        - Registration of the new function happens after the first match
    - Upon registration, we remove the match from the search space
    - If we see the same type again, it will work the same way as if we did things eagerly
    """

    __slots__ = ("_dispatcher",)
    _name: ClassVar[TranslateName]
    _dispatcher: SingleDispatchCallable[P, R_co]

    def __init__(self, f: Callable[P, R_co], /) -> None:
        # Trickery to steal the signature of the decorated function (for typing),
        # but overwriting the registry entry immediately
        # We need this as `self.__call__` has to reference `singledispatch.register`,
        # but a default implementation doesn't make much sense in this context
        dispatcher = singledispatch(f)
        dispatcher.register(object, self.__call__)
        self._dispatcher = dispatcher

    def __init_subclass__(cls, *, name: TranslateName, **_: Any) -> None:
        super().__init_subclass__()
        cls._name = name

    @staticmethod
    def is_native(native: Any, plugin: PluginAny) -> TypeIs[Incomplete]:
        raise NotImplementedError

    @staticmethod
    def native_classes(plugin: PluginAny) -> Iterator[type[Incomplete]]:
        raise NotImplementedError

    @classmethod
    def _find_plugin_or_raise(cls, native: Any) -> PluginAny:
        manager = PluginManager()
        parse = manager._plugin_parse
        native_kind = cls._name
        query = (
            p
            for p in manager._iter_plugins()
            if p.is_imported()
            and parse(p.name).has(native_kind)
            and cls.is_native(native, p)
        )
        if found := next(query, None):
            return found
        msg = f"Unsupported {native_kind} type, got: {qualified_type_name(native)!r}\n\n{native!r}"
        raise TypeError(msg)

    @classmethod
    def _rewrap_constructor(cls, plugin_name: PluginName) -> Callable[P, R_co]:
        """Allows the new constructor to be generic over `Version`."""
        manager = PluginManager()

        def from_native(*args: P.args, **kwds: P.kwargs) -> R_co:
            version: Version = kwds.pop("version")  # type: ignore[assignment]
            tp = manager._import_class(cls._name, plugin_name, version)
            compliant: R_co = tp.from_native(*args, **kwds)
            return compliant

        return from_native

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R_co:
        native = args[0]
        plugin = self._find_plugin_or_raise(native)
        constructor = self._rewrap_constructor(plugin.name)
        for tp_native in self.native_classes(plugin):
            self._dispatcher.register(tp_native, constructor)
        # Use that constructor, instead of calling back into `from_native_*` again
        return constructor(*args, **kwds)


# NOTE: These are here to provide a static connection back to `Plugin` methods
# fmt: off
class _FromNativeDataFrame(_FromNative[P, R_co], name="dataframe"):
    __slots__ = ()
    @staticmethod
    def is_native(native: Any, plugin: PluginAny) -> TypeIs[NativeDataFrame]:
        return plugin.is_native_dataframe(native)
    @staticmethod
    def native_classes(plugin: PluginAny) -> Iterator[type[NativeDataFrame]]:
        yield from plugin.native_dataframe_classes()
class _FromNativeLazyFrame(_FromNative[P, R_co], name="lazyframe"):
    __slots__ = ()
    @staticmethod
    def is_native(native: Any, plugin: PluginAny) -> TypeIs[NativeLazyFrame]:
        return plugin.is_native_lazyframe(native)
    @staticmethod
    def native_classes(plugin: PluginAny) -> Iterator[type[NativeLazyFrame]]:
        yield from plugin.native_lazyframe_classes()
class _FromNativeSeries(_FromNative[P, R_co], name="series"):
    __slots__ = ()
    @staticmethod
    def is_native(native: Any, plugin: PluginAny) -> TypeIs[NativeSeries]:
        return plugin.is_native_series(native)
    @staticmethod
    def native_classes(plugin: PluginAny) -> Iterator[type[NativeSeries]]:
        yield from plugin.native_series_classes()
# fmt: on


@_FromNativeDataFrame
def from_native_dataframe(native: DF, /, *, version: Version) -> ct.DataFrame[DF, Any]:
    raise NotImplementedError


@_FromNativeLazyFrame
def from_native_lazyframe(native: LF, /, *, version: Version) -> ct.LazyFrame[LF]:
    raise NotImplementedError


@_FromNativeSeries
def from_native_series(native: S, name: str, /, *, version: Version) -> ct.Series[S]:
    raise NotImplementedError

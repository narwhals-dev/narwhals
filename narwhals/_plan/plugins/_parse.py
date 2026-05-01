"""Moving a `Plugin` into a more understood representation.

## Notes
- Purely about getting the attributes accessible (for now)
- Overloads can happen further up
    - Trying to do them everywhere doesn't perform well
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, TypeVar, cast

from narwhals._plan._immutable import Immutable
from narwhals._plan.compliant import classes as cc
from narwhals._plan.typing import PluginAny, PluginName
from narwhals._utils import Version, deep_attrgetter

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan.compliant import typing as ct


Incomplete: TypeAlias = Any
PluginUnknown: TypeAlias = PluginAny
"""May want to give this *some* more detail later."""

_VersionName: TypeAlias = Literal["v1", "v2"]
_VERSIONS: Final[tuple[_VersionName, ...]] = "v1", "v2"
_ClassName: TypeAlias = Literal[
    "dataframe", "evaluator", "expr", "lazyframe", "scalar", "series"
]
_COMPLIANT_NAMES: Final = (
    "_dataframe",
    "_evaluator",
    "_expr",
    "_lazyframe",
    "_scalar",
    "_series",
)
_CompliantName: TypeAlias = Literal[
    "_dataframe", "_evaluator", "_expr", "_lazyframe", "_scalar", "_series"
]

UnsupportedName: TypeAlias = Literal[
    "DataFrame", "Series", "LazyFrame", "Expr", "Scalar", _VersionName
]
"""The name to display in an error message."""

R_co = TypeVar("R_co", covariant=True)
Accessor: TypeAlias = Callable[[cc.ClassesAny], R_co]
"""An inverted class accessor function.

These *take* a `__narwhals_classes__` as input, and (along the happy path)
return the requested class.

The first unique bit is that this can represent **any** version of any class, e.g.:

    __narwhals_classes__._dataframe
    __narwhals_classes__.v1._lazyframe
    __narwhals_classes__.v2._series

Second, is that if the class is not implemented (eager-only, no versioning, etc) the *"accessor"*
is a function that raises an exception:

    NotImplementedError: `LazyFrame()` is not supported for 'pyarrow'
"""


_REMAP_PROPERTIES: Final[Mapping[_ClassName, _CompliantName]] = {
    "dataframe": "_dataframe",
    "evaluator": "_evaluator",
    "expr": "_expr",
    "lazyframe": "_lazyframe",
    "scalar": "_scalar",
    "series": "_series",
}
_REMAP_ERRORS: Final[Mapping[_ClassName, UnsupportedName]] = {
    "dataframe": "DataFrame",
    "evaluator": "LazyFrame",
    "expr": "Expr",
    "lazyframe": "LazyFrame",
    "scalar": "Scalar",
    "series": "Series",
}


class ClassesProxyTD(TypedDict):
    """Each key mirrors a property on `*Classes`."""

    _dataframe: Accessor[type[ct.DataFrameAny]]
    _lazyframe: Accessor[type[ct.LazyFrameAny]]
    _evaluator: Accessor[type[ct.PlanEvaluatorAny]]
    _expr: Accessor[type[ct.ExprAny]]
    _scalar: Accessor[type[ct.ExprAny | ct.ScalarAny]]
    _series: Accessor[type[ct.SeriesAny]]


class RegEntry(TypedDict):
    """Versioned accessor functions, with error handling."""

    MAIN: ClassesProxyTD
    V1: ClassesProxyTD
    V2: ClassesProxyTD


PluginReg: TypeAlias = dict[PluginName, RegEntry]


class Unsupported:
    """Marker for functionality that should raise on use."""

    __slots__ = ("_feature", "_plugin")
    _feature: UnsupportedName
    _plugin: PluginName

    def __init__(self, feature: UnsupportedName, plugin: PluginName, /) -> None:
        self._feature = feature
        self._plugin = plugin

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._feature!r})"

    def __bool__(self) -> Literal[False]:
        return False

    def error(self) -> NotImplementedError:
        feature = self._feature
        if feature in _VERSIONS:
            msg = f"Version {feature!r} is not yet supported for {self._plugin!r}"
        else:
            msg = f"`{feature}()` is not supported for {self._plugin!r}"
        return NotImplementedError(msg)

    def __call__(self, obj: Any, /) -> Any:
        raise self.error()

    @staticmethod
    def fill_version(version: _VersionName, plugin: PluginName, /) -> ClassesProxyTD:
        m: Any = dict.fromkeys(_COMPLIANT_NAMES, Unsupported(version, plugin))
        r: ClassesProxyTD = m
        return r


class ClassesIR(Immutable):
    """A very simple representation for a single version's classes.

    We can discover this information *without* requiring dependencies.
    """

    __slots__ = ("dataframe", "evaluator", "expr", "lazyframe", "scalar", "series")
    expr: bool
    scalar: bool
    dataframe: bool
    series: bool
    lazyframe: bool
    evaluator: bool

    @staticmethod
    def from_classes(classes: cc.ClassesAny, /) -> ClassesIR:
        can_eager = cc.can_eager(classes)
        can_lazy = cc.can_lazy(classes)
        return ClassesIR(
            expr=True,
            scalar=True,
            dataframe=can_eager,
            series=can_eager,
            lazyframe=can_lazy,
            evaluator=can_lazy,
        )

    def to_accessors(
        self, plugin: PluginName, version: _VersionName | None = None
    ) -> ClassesProxyTD:
        """Convert this representation into a mapping of accessor functions."""
        props = _REMAP_PROPERTIES
        errors = _REMAP_ERRORS
        it = cast("Iterator[tuple[_ClassName, bool]]", self.__immutable_items__)
        prefix = () if version is None else (version,)
        results: Incomplete = {
            out_name: (
                deep_attrgetter(*prefix, out_name)
                if value
                else Unsupported(errors[name], plugin)
            )
            for name, value in it
            if (out_name := props[name])
        }
        out: ClassesProxyTD = results
        return out


class PluginIR(Immutable):
    """Feature flags for a plugin, documenting class/version support."""

    __slots__ = ("name", "main", "v1", "v2")  # noqa: RUF023
    name: PluginName
    main: ClassesIR
    v1: ClassesIR | Literal[False]
    v2: ClassesIR | Literal[False]

    @staticmethod
    def from_plugin(plugin: PluginUnknown, /) -> PluginIR:
        classes = plugin.__narwhals_classes__
        return PluginIR(
            name=plugin.name,
            main=ClassesIR.from_classes(classes),
            v1=(ClassesIR.from_classes(classes.v1) if cc.can_v1(classes) else False),
            v2=(ClassesIR.from_classes(classes.v2) if cc.can_v2(classes) else False),
        )

    def to_registry_item(self) -> tuple[PluginName, RegEntry]:
        plugin = self.name
        accessors: RegEntry = {
            "MAIN": self.main.to_accessors(plugin),
            "V1": (
                v1.to_accessors(plugin, "v1")
                if (v1 := self.v1)
                else Unsupported.fill_version("v1", plugin)
            ),
            "V2": (
                v2.to_accessors(plugin, "v2")
                if (v2 := self.v2)
                else Unsupported.fill_version("v2", plugin)
            ),
        }
        return self.name, accessors


# TODO @dangotbanned: Call this from `PluginManager`
parse_plugin: Final = PluginIR.from_plugin


# TODO @dangotbanned: This would be handled by `PluginManager`
def _get_from_plugin(
    plugin_name: PluginName, attr_name: _CompliantName, version: Version
) -> Incomplete:
    """The context the query starts from.

    The goal for `PluginIR` is to create something that is easy to use from this.
    """
    from narwhals._plan.plugins._manager import PluginManager

    manager = PluginManager()
    plugin = manager.get(plugin_name, require="is_imported")
    registry = manager._registry
    return registry[plugin_name][version.name][attr_name](plugin.__narwhals_classes__)  # type: ignore[literal-required]

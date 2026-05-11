"""Moving a `Plugin` into a more understood representation.

## Notes
- Purely about getting the attributes accessible (for now)
- Overloads can happen further up
    - Trying to do them everywhere doesn't perform well
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypedDict,
    TypeVar,
    cast,
    get_args,
)

from narwhals._plan._immutable import Immutable
from narwhals._plan.compliant import classes as cc
from narwhals._utils import Version, deep_attrgetter

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import LiteralString, TypeAlias

    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.typing import PluginAny, PluginName, Seq, VersionName


Incomplete: TypeAlias = Any
R_co = TypeVar("R_co", covariant=True)
Accessor: TypeAlias = Callable[[cc.ClassesAny], R_co]
"""An inverted class accessor function."""

TranslateName: TypeAlias = Literal["dataframe", "lazyframe", "series"]


class ClassesProxyTD(TypedDict):
    """Each key mirrors a property on `*Classes`."""

    dataframe: Accessor[type[ct.DataFrameAny]]
    lazyframe: Accessor[type[ct.LazyFrameAny]]
    evaluator: Accessor[type[ct.PlanEvaluatorAny]]
    expr: Accessor[type[ct.ExprAny]]
    scalar: Accessor[type[ct.ExprAny | ct.ScalarAny]]
    series: Accessor[type[ct.SeriesAny]]


class RegEntry(TypedDict):
    """A versioned mapping of class accessors.

    Each `Accessor` is a function that can return a specific class:

        expr: Accessor[type[ct.ExprAny]]
        got = type[ct.ExprAny] = expr(plugin.__narwhals_classes__)

    The structure of each entry provides a way to parametrize the version and class:

        classes: cc.ClassesAny = plugin.__narwhals_classes__
        entry: RegEntry                     # Equivalent to
        entry["MAIN"]["dataframe"](classes) # -> classes.dataframe
        entry["V1"]["lazyframe"](classes)   # -> classes.v1.lazyframe
        entry["V2"]["series"](classes)      # -> classes.v2.series

    If the class is not implemented the `Accessor` is a function that raises an exception:

        NotImplementedError: `LazyFrame()` is not supported for 'pyarrow'.

    All together this avoids requiring the caller to repeatedly check:
    - *do we have a `v1`?*
    - *do we have a `dataframe`?*

    Because we asked that question *already* and prepared our response 😅
    """

    MAIN: ClassesProxyTD
    V1: ClassesProxyTD
    V2: ClassesProxyTD


class Unsupported(Immutable):
    """Marker for functionality that should raise on use."""

    __slots__ = ("feature", "plugin")
    feature: cc.PropertyName | VersionName
    plugin: PluginName
    __repr__ = Immutable.__str__
    _REMAP_ERRORS: ClassVar[Mapping[cc.PropertyName | VersionName, LiteralString]] = {
        "dataframe": "DataFrame",
        "evaluator": "LazyFrame",
        "expr": "Expr",
        "lazyframe": "LazyFrame",
        "scalar": "Scalar",
        "series": "Series",
        "MAIN": "MAIN",
        "V1": "V1",
        "V2": "V2",
    }
    """The name to display in an error message."""

    def error(self) -> NotImplementedError:  # pragma: no cover
        feature = self._REMAP_ERRORS[self.feature]
        if feature in Version._member_names_:
            msg = f"Version {feature!r} is not yet supported for {self.plugin!r}"
        else:
            msg = f"`{feature}()` is not supported for {self.plugin!r}"
        return NotImplementedError(msg)

    def __call__(self, obj: Any, /) -> Any:  # pragma: no cover
        raise self.error()

    @staticmethod
    def fill_version(
        version: VersionName, plugin: PluginName, /
    ) -> ClassesProxyTD:  # pragma: no cover
        obj = Unsupported(feature=version, plugin=plugin)
        m: Any = dict.fromkeys(get_args(cc.PropertyName), obj)
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
    __repr__ = Immutable.__str__

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
        self, plugin: PluginName, version: VersionName | None = None
    ) -> ClassesProxyTD:
        """Convert this representation into a mapping of accessor functions."""
        it = cast("Iterator[tuple[cc.PropertyName, bool]]", self.__immutable_items__)
        prefix = () if version is None else (version.lower(),)
        getter = deep_attrgetter
        results: Incomplete = {
            feature: (
                getter(*prefix, feature)
                if has
                else Unsupported(feature=feature, plugin=plugin)
            )
            for feature, has in it
        }
        out: ClassesProxyTD = results
        return out


class PluginIR(Immutable):
    """Feature flags for a plugin, documenting class/version support."""

    __slots__ = ("name", "main", "v1", "v2", "translate")  # noqa: RUF023
    name: PluginName
    main: ClassesIR
    v1: ClassesIR | Literal[False]
    v2: ClassesIR | Literal[False]
    translate: frozenset[TranslateName]
    __repr__ = Immutable.__str__

    def has(self, native_name: TranslateName) -> bool:
        """Return True if the plugin handles `native_name` in `from_native`."""
        return native_name in self.translate

    @staticmethod
    def from_plugin(plugin: PluginAny, /) -> PluginIR:
        classes = plugin.__narwhals_classes__
        main = ClassesIR.from_classes(classes)
        _names: Seq[TranslateName] = ("dataframe", "lazyframe", "series")
        return PluginIR(
            name=plugin.name,
            main=main,
            v1=(ClassesIR.from_classes(classes.v1) if cc.can_v1(classes) else False),
            v2=(ClassesIR.from_classes(classes.v2) if cc.can_v2(classes) else False),
            translate=frozenset(name for name in _names if getattr(main, name)),
        )

    def to_registry_entry(self) -> RegEntry:
        plugin = self.name
        return {
            "MAIN": self.main.to_accessors(plugin),
            "V1": (
                v1.to_accessors(plugin, "V1")
                if (v1 := self.v1)
                else Unsupported.fill_version("V1", plugin)
            ),
            "V2": (
                v2.to_accessors(plugin, "V2")
                if (v2 := self.v2)
                else Unsupported.fill_version("V2", plugin)
            ),
        }

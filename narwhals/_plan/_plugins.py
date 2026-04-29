"""Managing plugins."""

from __future__ import annotations

import functools
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeVar, overload

from narwhals._plan.compliant import classes as cc
from narwhals._plan.exceptions import unsupported_error
from narwhals._typing_compat import assert_never
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.metadata import EntryPoint, EntryPoints

    from typing_extensions import Never, TypeAlias

    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.compliant.classes import (
        ClassesT_co as C,
        ClassesV1T_co as C1,
        ClassesV2T_co as C2,
    )
    from narwhals._plan.compliant.plugins import BuiltinAny, Plugin, PluginAny, PluginName
    from narwhals._plan.compliant.typing import (
        DataFrameAny,
        DataFrameT_co as DF,
        EagerDataFrameT_co as EagerDF,
        ExprT_co as E,
        LazyFrameT_co as LF,
        PlanEvaluatorT_co as PE,
        ScalarNoDefaultT_co as SC,
        SeriesT_co as S,
    )
    from narwhals._plan.plans.visitors import ResolvedToCompliantAny as PlanEvaluatorAny
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.typing import BackendTodo, NativeModuleType
    from narwhals._typing import Arrow, BackendName, Polars
    from narwhals.typing import Backend, IntoBackend

Incomplete: TypeAlias = Any

TemporaryPluginsType: TypeAlias = "dict[str, BuiltinAny | PluginAny]"
"""Obviously don't want to keep this for long"""

IntoBackendExt: TypeAlias = "IntoBackend[Backend] | PluginName | Implementation"

_UNKNOWN: Final = Implementation.UNKNOWN

_R_co = TypeVar("_R_co", covariant=True)


class _Plugin(Protocol[_R_co]):
    """Minimal interface for typing.

    Doesn't set a bound for `__narwhals_classes__`.
    """

    @property
    def plugin_name(self) -> PluginName: ...
    @property
    def __narwhals_classes__(self) -> _R_co: ...


_PluginV1: TypeAlias = _Plugin[cc.HasV1["C1"]]
_PluginV2: TypeAlias = _Plugin[cc.HasV2["C2"]]


# TODO @dangotbanned: (low-priority) Remove 3.10 guard after https://github.com/narwhals-dev/narwhals/issues/3204
# TODO @dangotbanned: (low-priority) Cover the duplicate name plugin case
@functools.cache
def _entry_points() -> EntryPoints:
    # NOTE: Wrappped with some one-time validation, so everything outside is simpler
    from importlib.metadata import entry_points

    if sys.version_info < (3, 10):
        msg = "Need `EntryPoints.{select,names}`, this can wait until 3.10 "
        raise NotImplementedError(msg)
    group = "narwhals.plugins-plan"
    if (eps := entry_points(group=group)) and len(eps) == len(eps.names):
        return eps
    if not eps:  # pragma: no cover
        # If you're developing narwhals, this may have failed due to the `group` being renamed,
        # see `[project.entry-points.<group>]` in pyproject.toml
        call = f"{entry_points.__qualname__}(group={group!r})"
        msg = f"Expected to find built-in backends, but `{call}`\nreturned {eps!r}"
        raise NotImplementedError(msg)
    msg = f"Multiple plugins found with the same `name`:\n{eps!r}"  # pragma: no cover
    raise NotImplementedError(msg)


def _load_entry_point(entry_point: EntryPoint, /) -> PluginAny | BuiltinAny:
    """Use this to add the typing in a consistent way.

    May need to iterate on what the initial type(s) are some more.
    """
    plugin: PluginAny | BuiltinAny = entry_point.load()
    return plugin


# TODO @dangotbanned: Figure out what self-entry points are needed
def _load_plugins() -> Iterator[tuple[str, PluginAny | BuiltinAny]]:
    """Load all entry points.

    ## Notes
    - what does a `Plugin` manager look like?
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
    for ep in _entry_points():
        yield ep.name, _load_entry_point(ep)


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
    name = _backend_to_plugin_name(backend)
    eps = _entry_points()
    if found := eps.select(name=name):
        return _load_entry_point(next(iter(found)))
    raise _unsupported_error(backend, name, eps)


# TODO @dangotbanned: Replace ASAP
def _get_plugin_importable(
    plugins: TemporaryPluginsType, backend: IntoBackendExt, /
) -> PluginAny | BuiltinAny:
    # NOTE:  this will be a method of "plugins"
    name = _backend_to_plugin_name(backend)
    if (plugin := plugins.get(name)) is None:
        raise _unsupported_error(backend, name, _entry_points())  # pragma: no cover
    if not plugin.can_import():
        raise _unavailable_error(plugin)  # pragma: no cover
    return plugin


def import_evaluator(
    plugin: Plugin[cc.LazyClasses[LF, PE, E, SC], Any, Any, Any] | PluginAny,
    version: Version,
) -> type[PE]:
    # NOTE: Seemingly the only thing mypy is fine with
    # But it forgets what this means as soon as you leave the function 😭
    classes = import_classes(plugin, version)
    if cc.can_lazy(classes):
        return classes._evaluator
    raise unsupported_error(plugin.plugin_name, "LazyFrame.collect")  # pragma: no cover


def import_dataframe(
    plugin: Plugin[cc.EagerClasses[DF | EagerDF, S, E, SC], Any, Any, Any] | PluginAny,
    version: Version,
) -> type[DF | EagerDF]:
    classes = import_classes(plugin, version)
    if cc.can_eager(classes):
        return classes._dataframe
    raise unsupported_error(plugin.plugin_name, "DataFrame")  # pragma: no cover


@overload
def import_classes(plugin: _Plugin[C], version: Literal[Version.MAIN]) -> C: ...
@overload
def import_classes(plugin: _PluginV1[C1], version: Literal[Version.V1]) -> C1: ...
@overload
def import_classes(plugin: _PluginV2[C2], version: Literal[Version.V2]) -> C2: ...
@overload
def import_classes(
    plugin: _Plugin[C] | _PluginV1[C1] | _PluginV2[C2], version: Version
) -> C | C1 | C2: ...
def import_classes(
    plugin: _Plugin[C] | _PluginV1[C1] | _PluginV2[C2], version: Version
) -> Incomplete:
    """Import the accessor to the compliant classes compatible with `version`."""
    classes = plugin.__narwhals_classes__
    if version is Version.MAIN:
        return classes
    if version is Version.V1:
        if cc.can_v1(classes):
            return classes.v1
        raise unsupported_error(plugin.plugin_name, "v1")  # pragma: no cover
    if version is Version.V2:
        if cc.can_v2(classes):
            return classes.v2
        raise unsupported_error(plugin.plugin_name, "v2")  # pragma: no cover
    assert_never(version)


# TODO @dangotbanned: (low-priority) get the optional `Resolver` using this backend
def lazyframe_collect(
    current_backend: IntoBackendExt,
    collect_backend: IntoBackendExt | None = None,
    version: Version = Version.MAIN,
) -> tuple[type[PlanEvaluatorAny], type[DataFrameAny]]:
    """Mocking this call as it does some gymnastics I'd like to avoid.

    Important:
        Pretending that we have zero plugins loaded *for now*, need to start somewhere

    ## Overview of current
    - [`collect`] and [`sink_parquet`] both are acrobats
    - `Resolver.from_backend` does a second call to `known_implementation` on `current_implementation` (lazy)
        - Starting the plan would've validated at least once already
    - `PolarsEvaluator.collect` does `(backend or "polars")` before ...
        - `CompliantLazyFrame.collect_compliant` does a second validation on `collect_backend` (backend)

    [`collect`]: https://github.com/narwhals-dev/narwhals/blob/be25d3fdd96a51aad08f513d5e45e41703960c49/narwhals/_plan/lazyframe.py#L312-L320
    [`sink_parquet`]: https://github.com/narwhals-dev/narwhals/blob/be25d3fdd96a51aad08f513d5e45e41703960c49/narwhals/_plan/lazyframe.py#L322-L326
    """
    if current_backend is _UNKNOWN:  # pragma: no cover
        msg = (
            "Storing an unknown implementation on `LazyFrame` cannot work with plugins.\n"
            "`LazyFrame` needs a connection back to the plugin that handles the native object (e.g. `plugin_name`)."
        )
        raise TypeError(msg)

    plugins = dict(_load_plugins())
    lazy = _get_plugin_importable(plugins, current_backend)
    evaluator: type[PlanEvaluatorAny] = import_evaluator(lazy, version)
    eager = _get_plugin_importable(plugins, collect_backend) if collect_backend else lazy
    # NOTE: Annotating this to please `mypy` prevents `pyright` from inferring ` type[PolarsDataFrame] | type[ArrowDataFrame]`
    dataframe = import_dataframe(eager, version)  # type: ignore[var-annotated]
    return evaluator, dataframe


def _backend_to_plugin_name(
    backend: IntoBackendExt, /
) -> BackendName | PluginName:  # pragma: no cover
    if isinstance(backend, str):
        return backend
    if backend is _UNKNOWN or (impl := Implementation.from_backend(backend)) is _UNKNOWN:
        msg = f"{_UNKNOWN!r} is not supported in this context, got: {backend!r}"
        raise NotImplementedError(msg)
    name: BackendName = impl.value
    return name


def _unsupported_error(backend: Any, name: str, eps: EntryPoints, /) -> Exception:
    if (impl := Implementation.from_backend(name)) is not _UNKNOWN:
        msg = f"{impl!r} is not yet supported in `narwhals._plan`, got: {backend!r}"
        return NotImplementedError(msg)
    msg = f"Unsupported `backend` value.\nExpected one of {sorted(eps.names)!r}, got: {backend!r}"
    return TypeError(msg)


def _unavailable_error(plugin: PluginAny) -> Exception:  # pragma: no cover
    missing = [name for name in plugin.sys_modules_targets if find_spec(name) is None]
    msg = f"Plugin {plugin.plugin_name!r} was found but could not import the following required modules: {missing!r}"
    return ModuleNotFoundError(msg)

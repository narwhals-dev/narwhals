from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, Literal

import pytest

from narwhals._plan._plugins import PluginManager, lazyframe_collect
from narwhals._plan.plugins import load_plugin
from narwhals._typing import Arrow, Polars
from narwhals._typing_compat import assert_never
from narwhals._utils import Implementation, Version
from tests.plan.utils import re_compile

if TYPE_CHECKING:
    from collections.abc import Generator

    from typing_extensions import LiteralString, TypeAlias, assert_type

    from narwhals._plan import _plugins
    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.compliant.typing import DataFrameAny, PlanEvaluatorAny
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.typing import BuiltinAny, IntoBackendExt, PluginAny
    from narwhals.typing import Backend, EagerAllowed, IntoBackend, LazyAllowed


SupportedBackend: TypeAlias = Literal[Arrow, Polars]
BuiltinName: TypeAlias = Literal["polars", "pyarrow"]
BUILTIN_NAMES = ("polars", "pyarrow")


MYPY: Final = False
"""Maybe one day mypy will understand better.

Currently just aiming for not getting `Never` everywhere.
"""


@pytest.fixture
def plugin(eager: BuiltinName) -> Generator[BuiltinAny, Any, None]:
    """Yields the result of `load_plugin(eager)`.

    We use a context manager to first clear and then reset (related) changes to `sys.modules`.
    """
    with pytest.MonkeyPatch.context() as mp:
        for name in BUILTIN_NAMES:
            mp.delitem(sys.modules, name, raising=False)
        yield load_plugin(eager)


def test_load_builtin(eager: BuiltinName) -> None:
    plugin = load_plugin(eager)
    assert plugin.name == eager
    assert plugin.implementation is Implementation.from_backend(eager)


def test_load_plugin_invalid() -> None:
    with pytest.raises(NotImplementedError, match=r"not yet"):
        assert_never(load_plugin("modin"))
    with pytest.raises(
        TypeError, match=re_compile(r"Unsupported `backend` .+got: 'i dont exist'")
    ):
        load_plugin("i dont exist")


def test_plugin_is_imported(plugin: BuiltinAny) -> None:
    assert not plugin.is_imported()
    PluginManager().import_modules(plugin.name)

    assert plugin.is_imported()
    assert load_plugin(plugin.name).is_imported()


def test_plugin_can_import(plugin: BuiltinAny) -> None:
    assert not plugin.is_imported()
    assert plugin.can_import()

    PluginManager().import_modules(plugin.name)
    assert plugin.is_imported()
    assert plugin.can_import()
    assert load_plugin(plugin.name).can_import()

    PluginManager().import_modules(plugin.name)
    assert plugin.can_import()


def test_plugin_manager_known() -> None:
    assert sorted(PluginManager().known()) == ["polars", "pyarrow"]


def test_plugin_manager_importable() -> None:
    plug_man = PluginManager()
    for name in plug_man.importable():
        plug_man.import_modules(name)
        assert plug_man.get(name).is_imported()


def test_plugin_manager_imported(plugin: BuiltinAny) -> None:
    plug_man = PluginManager()
    name = plugin.name
    assert name not in set(plug_man.imported())
    plug_man.import_modules(name)
    assert name in set(plug_man.imported())


# TODO @dangotbanned: Replace with something less experimental (when available)
@pytest.mark.parametrize("version", Version)
def test_lazyframe_collect(eager: EagerAllowed, version: Version) -> None:
    """WIP, not a real API!"""
    pytest.importorskip("polars")
    current = "polars"
    evaluator, dataframe = lazyframe_collect(current, eager, version)
    assert evaluator.implementation.is_polars()
    assert dataframe.implementation is Implementation.from_backend(eager)
    assert dataframe.version is evaluator.version is version


if TYPE_CHECKING:
    import random
    from types import ModuleType

    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from narwhals._plan import arrow as pa_main, polars as pl_main
    from narwhals._plan.arrow import v1 as pa_v1, v2 as pa_v2
    from narwhals._plan.arrow.classes import ArrowClasses, ArrowClassesV1, ArrowClassesV2
    from narwhals._plan.compliant import classes as cc
    from narwhals._plan.polars import v1 as pl_v1, v2 as pl_v2
    from narwhals._plan.polars.classes import (
        PolarsClasses,
        PolarsClassesV1,
        PolarsClassesV2,
    )

    def typing_load_plugin(
        never_builtin: LiteralString,
        always_builtin_1: ModuleType,
        always_builtin_2: SupportedBackend,
        always_polars: Polars,
        always_pyarrow: Arrow,
        too_dynamic: str,
        not_yet_1: IntoBackend[Backend],
        not_yet_2: EagerAllowed,
        not_yet_3: LazyAllowed,
    ) -> None:
        # NOTE: Purely checking the result of matching `@overload`s
        assert_type(load_plugin(never_builtin), PluginAny)
        assert_type(load_plugin(always_builtin_1), BuiltinAny)
        assert_type(load_plugin(always_builtin_2), BuiltinAny)
        assert_type(load_plugin(always_polars), PolarsPlugin)
        assert_type(load_plugin(always_pyarrow), ArrowPlugin)

        # Until `mypy` supports PEP 675 it won't report anything
        # https://github.com/python/mypy/issues/12554
        load_plugin(too_dynamic)  # pyright: ignore[reportArgumentType, reportCallIssue]
        load_plugin(not_yet_1)  # pyright: ignore[reportArgumentType, reportCallIssue]
        load_plugin(not_yet_2)  # pyright: ignore[reportArgumentType, reportCallIssue]
        load_plugin(not_yet_3)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def typing_plugin_guards(
        df_polars: pl.DataFrame, df_pyarrow: pa.Table, df_pandas: pd.DataFrame
    ) -> None:
        # NOTE: If we've matched the right `@overload`, the type guards for each plugin should allow us to narrow
        # in different ways, depending on how the type parameter for the method resolved
        non_static = random.choice([True, False])  # noqa: S311

        polars_or_pyarrow = df_polars if non_static else df_pyarrow
        polars_or_pandas = df_polars if non_static else df_pandas
        pyarrow_or_pandas = df_pyarrow if non_static else df_pandas
        who_knows = pyarrow_or_pandas if non_static else polars_or_pyarrow

        plugin = load_plugin("polars")

        if plugin.is_native_dataframe(df_polars):
            ...
        else:
            assert_never(df_polars)

        if plugin.is_native_lazyframe(df_polars):
            # NOTE: Tests that the guard produces an intersection, rather than forcing `DataFrame` -> `LazyFrame`
            #   "assert_type" mismatch: expected "LazyFrame" but received "<subclass of DataFrame and LazyFrame>"
            # Can't represent this yet
            # https://github.com/python/typing/issues/213
            # https://github.com/CarliJoy/intersection_examples/issues/53
            assert_type(df_polars, pl.LazyFrame)  # pyright: ignore[reportAssertTypeFailure]

        if not plugin.is_native_dataframe(polars_or_pandas):
            assert_type(polars_or_pandas, pd.DataFrame)

        if not plugin.is_native_dataframe(who_knows):
            if not load_plugin(Implementation.PYARROW).is_native_dataframe(who_knows):
                assert_type(who_knows, pd.DataFrame)
            else:
                assert_type(who_knows, pa.Table)
        else:
            assert_type(who_knows, pl.DataFrame)

    # TODO @dangotbanned: Would be nice for this to preserve the implementation
    def typing_plugin_dogfood_implementation() -> None:
        polars_1 = load_plugin("polars").implementation
        polars_2 = load_plugin(polars_1)  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert_type(polars_2, Literal[Implementation.POLARS])  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure]

        pyarrow_1 = load_plugin("polars").implementation
        pyarrow_2 = load_plugin(pyarrow_1)  # pyright: ignore[reportArgumentType, reportCallIssue]
        assert_type(pyarrow_2, Literal[Implementation.PYARROW])  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure]

    def typing_can_eager_lazy_integration(
        current: IntoBackendExt, collect: IntoBackendExt | None, version: Version
    ) -> tuple[type[PlanEvaluatorAny], type[DataFrameAny]]:
        """By far the most insane idea yet.

        - This took an incredibly long time to get (mostly) working
        - Assertions are more detailed that usual
        - Need this here while I try to minimize the typing
        """
        plugins = PluginManager()
        lazy = plugins.get(current)
        assert_type(lazy, PluginAny | BuiltinAny)
        classes_1 = _plugins.import_classes(lazy, version)

        if MYPY:
            assert_type(classes_1, Any)
        else:
            assert_type(
                classes_1,
                PolarsClasses
                | ArrowClasses
                | PolarsClassesV1
                | PolarsClassesV2
                | ArrowClassesV1
                | ArrowClassesV2
                | Any,
            )

        if cc.can_lazy(classes_1):
            evaluator = classes_1._evaluator
            if MYPY:
                assert_type(evaluator, type[Any])
            else:
                assert_type(
                    evaluator,
                    type[
                        pl_main.PlanEvaluator | pl_v1.PlanEvaluator | pl_v2.PlanEvaluator
                    ],
                )
        else:
            raise NotImplementedError

        eager = plugins.get(collect) if collect else lazy
        assert_type(eager, PluginAny | BuiltinAny)

        classes_2 = _plugins.import_classes(eager, version)
        if MYPY:
            assert_type(classes_2, Any)
        else:
            assert_type(
                classes_2,
                PolarsClasses
                | ArrowClasses
                | PolarsClassesV1
                | PolarsClassesV2
                | ArrowClassesV1
                | ArrowClassesV2
                | Any,
            )

        if cc.can_eager(classes_2):
            dataframe = classes_2._dataframe
            if MYPY:
                assert_type(dataframe, type[Any])
            else:
                assert_type(
                    dataframe,
                    type[
                        pa_main.DataFrame
                        | pa_v1.DataFrame
                        | pa_v2.DataFrame
                        | pl_main.DataFrame
                        | pl_v1.DataFrame
                        | pl_v2.DataFrame
                    ],
                )

        else:
            raise NotImplementedError
        return evaluator, dataframe

    def typing_import_classes(
        builtin: BuiltinAny,
        unknown: PluginAny,
        builtin_unknown: BuiltinAny | PluginAny,
        pyarrow_unknown: ArrowPlugin | PluginAny,
        polars_unknown: PolarsPlugin | PluginAny,
        version: Version,
        main: Literal[Version.MAIN],
        v1: Literal[Version.V1],
        v2: Literal[Version.V2],
    ) -> None:
        # NOTE: Has nothing to match on
        unknown_any_version = _plugins.import_classes(unknown, version)  # type: ignore[var-annotated]
        unknown_main = _plugins.import_classes(unknown, main)
        unknown_v1 = _plugins.import_classes(unknown, v1)
        unknown_v2 = _plugins.import_classes(unknown, v2)
        assert_type(unknown_any_version, Any)
        assert_type(unknown_main, Any)
        assert_type(unknown_v1, Any)
        assert_type(unknown_v2, Any)

        # NOTE: We know all classes will be builtin, just need to narrow on version
        builtin_any_version = _plugins.import_classes(builtin, version)
        builtin_main = _plugins.import_classes(builtin, main)
        builtin_v1 = _plugins.import_classes(builtin, v1)
        builtin_v2 = _plugins.import_classes(builtin, v2)
        assert_type(
            builtin_any_version,
            PolarsClasses
            | ArrowClasses
            | PolarsClassesV1
            | PolarsClassesV2
            | ArrowClassesV1
            | ArrowClassesV2,
        )
        assert_type(builtin_main, PolarsClasses | ArrowClasses)
        assert_type(builtin_v1, PolarsClassesV1 | ArrowClassesV1)
        assert_type(builtin_v2, PolarsClassesV2 | ArrowClassesV2)

        # NOTE: Mixing an unknown with all or a subset of builtins should simply add `Any`
        builtin_unknown_any_version = _plugins.import_classes(builtin_unknown, version)
        pyarrow_unknown_any_version = _plugins.import_classes(pyarrow_unknown, version)
        polars_unknown_any_version = _plugins.import_classes(polars_unknown, version)
        # NOTE: This first one is the most important, as it reflects `(PluginAny | BuiltinAny, Version)`
        assert_type(
            builtin_unknown_any_version,
            PolarsClasses
            | ArrowClasses
            | PolarsClassesV1
            | PolarsClassesV2
            | ArrowClassesV1
            | ArrowClassesV2
            | Any,
        )
        assert_type(
            pyarrow_unknown_any_version,
            ArrowClasses | ArrowClassesV1 | ArrowClassesV2 | Any,
        )
        assert_type(
            polars_unknown_any_version,
            PolarsClasses | PolarsClassesV1 | PolarsClassesV2 | Any,
        )

        # TODO @dangotbanned: Preserve the `Any` for unknown
        builtin_unknown_main = _plugins.import_classes(builtin_unknown, main)
        builtin_unknown_v1 = _plugins.import_classes(builtin_unknown, v1)
        builtin_unknown_v2 = _plugins.import_classes(builtin_unknown, v2)
        assert_type(builtin_unknown_main, PolarsClasses | ArrowClasses | Any)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(builtin_unknown_v1, PolarsClassesV1 | ArrowClassesV1 | Any)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(builtin_unknown_v2, PolarsClassesV2 | ArrowClassesV2 | Any)  # pyright: ignore[reportAssertTypeFailure]

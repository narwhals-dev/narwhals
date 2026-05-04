from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.compliant.typing as ct
from narwhals._plan.plugins import load_plugin, manager
from narwhals._typing import Arrow, Polars
from narwhals._typing_compat import assert_never
from narwhals._utils import Implementation, Version
from tests.plan.utils import re_compile

if TYPE_CHECKING:
    from collections.abc import Generator

    from typing_extensions import LiteralString, TypeAlias, assert_type

    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.compliant import CompliantDataFrame
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._plan.typing import BuiltinAny, IntoPlugin, PluginAny
    from narwhals.typing import Backend, EagerAllowed, IntoBackend, LazyAllowed

MYPY: Final = False

SupportedBackend: TypeAlias = Literal[Arrow, Polars]
BuiltinName: TypeAlias = Literal["polars", "pyarrow"]
BUILTIN_NAMES = ("polars", "pyarrow")


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
    manager().import_modules(plugin.name)

    assert plugin.is_imported()
    assert manager().plugin(plugin.name).is_imported()


def test_plugin_can_import(plugin: BuiltinAny) -> None:
    assert not plugin.is_imported()
    assert plugin.can_import()

    manager().import_modules(plugin.name)
    assert plugin.is_imported()
    assert plugin.can_import()
    assert load_plugin(plugin.name).can_import()

    manager().import_modules(plugin.name)
    assert plugin.can_import()


def test_plugin_manager_known() -> None:
    assert sorted(manager().known()) == ["polars", "pyarrow"]


def test_plugin_manager_importable() -> None:
    plug_man = manager()
    for name in plug_man.importable():
        plug_man.import_modules(name)
        assert plug_man.plugin(name).is_imported()


def test_plugin_manager_imported(plugin: BuiltinAny) -> None:
    plug_man = manager()
    name = plugin.name
    assert name not in set(plug_man.imported())
    plug_man.import_modules(name)
    assert name in set(plug_man.imported())


def test_plugin_manager_is_native_dataframe(eager: BuiltinName) -> None:
    plug_man = manager()
    data = {"a": [1, 2, 3]}
    assert not plug_man.is_native_dataframe(data)
    plugin = plug_man.plugin(eager)
    assert plugin.can_import()
    compliant = plug_man.dataframe(plugin.name, Version.MAIN).from_dict(data)
    assert not plug_man.is_native_dataframe(compliant)
    native = compliant.native
    assert plug_man.is_native_dataframe(native)
    assert plugin.is_native_dataframe(native)
    assert manager().is_native_dataframe(native)


@pytest.mark.parametrize("version", Version)
def test_plugin_manager_dataframe(eager: BuiltinName, version: Version) -> None:
    data = {"a": [1, 2, 3, 4], "b": [1.3, 1.9, None, None]}
    schema = {"a": nw.UInt32(), "b": nw.Float32()}
    impl = Implementation.from_backend(eager)

    compliant_type = manager().dataframe(eager, version)
    compliant = compliant_type.from_dict(data, schema=schema)
    nw_ = compliant.to_narwhals()
    nw_2 = manager().dataframe(impl, version).from_dict(data, schema=schema).to_narwhals()
    assert_context_preserved(compliant_type, compliant, nw_, nw_2, impl, version)


@pytest.mark.parametrize("version", Version)
def test_plugin_manager_series(eager: BuiltinName, version: Version) -> None:
    data = [1, 2, 3, 4]
    impl = Implementation.from_backend(eager)

    compliant_type = manager().series(eager, version)
    compliant = compliant_type.from_iterable(data, name="hello")
    nw_ = compliant.to_narwhals()
    nw_2 = manager().series(impl, version).from_iterable(data, name="hello").to_narwhals()
    assert_context_preserved(compliant_type, compliant, nw_, nw_2, impl, version)


@pytest.mark.parametrize("version", Version)
def test_plugin_manager_lazyframe(lazy: LazyAllowed, version: Version) -> None:
    _pa = pytest.importorskip("pyarrow")
    data = {"a": [1, 2, 3, 4], "b": [1.3, 1.9, None, None]}
    table: pa.Table = _pa.table(data)
    impl = Implementation.from_backend(lazy)

    compliant_type = manager().lazyframe(lazy, version)
    compliant = compliant_type.from_arrow(table)
    nw_ = compliant.collect_narwhals("pyarrow").lazy(lazy)
    nw_2 = (
        manager()
        .lazyframe(impl, version)
        .from_arrow(table)
        .collect_narwhals("pyarrow")
        .lazy(lazy)
    )
    assert_context_preserved(compliant_type, compliant, nw_, nw_2, impl, version)


CompT = TypeVar("CompT", bound="ct.DataFrameAny | ct.SeriesAny | ct.LazyFrameAny")
NwT = TypeVar(
    "NwT", bound="nwp.DataFrame[Any, Any] | nwp.Series[Any] | nwp.LazyFrame[Any]"
)


def assert_context_preserved(
    compliant_type: type[CompT],
    compliant_instance: CompT,
    narwhals_instance_1: NwT,
    narwhals_instance_2: NwT,
    implementation: Implementation,
    version: Version,
) -> None:
    """Ensure all arguments share implementations & versions."""
    assert compliant_type.version is version
    assert compliant_instance.version is version
    assert narwhals_instance_1.version is version
    assert narwhals_instance_2.version is version

    assert compliant_type.implementation is implementation
    assert compliant_instance.implementation is implementation
    assert narwhals_instance_1.implementation is implementation
    assert narwhals_instance_2.implementation is implementation

    assert type(narwhals_instance_1) is type(narwhals_instance_2)


@pytest.mark.xfail(
    reason="TODO @dangotbanned: Missing version(s); missing requirements; duplicate name; failed entry point load, etc",
    raises=NotImplementedError,
)
def test_mock_plugins() -> None:
    raise NotImplementedError


@pytest.mark.parametrize("version", Version)
def test_plugin_manager_evaluator(lazy: LazyAllowed, version: Version) -> None:
    # NOTE: Covered more thoroughly inside `LazyFrame`
    impl = Implementation.from_backend(lazy)
    compliant_1 = manager().evaluator(lazy, version)
    compliant_2 = manager().evaluator(impl.to_native_namespace(), version)

    assert compliant_1.version is version
    assert compliant_2.version is version

    assert compliant_1.implementation is impl
    assert compliant_2.implementation is impl


if TYPE_CHECKING:
    import random
    from types import ModuleType

    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from narwhals._plan import arrow as pa_main, polars as pl_main
    from narwhals._plan.arrow import v1 as pa_v1, v2 as pa_v2
    from narwhals._plan.arrow.classes import ArrowClasses
    from narwhals._plan.compliant import classes as cc
    from narwhals._plan.polars import v1 as pl_v1, v2 as pl_v2
    from narwhals._plan.polars.classes import PolarsClasses

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
        assert_type(manager().plugin(never_builtin), PluginAny)
        assert_type(manager().plugin(always_builtin_1), BuiltinAny)
        assert_type(manager().plugin(always_builtin_2), BuiltinAny)
        assert_type(manager().plugin(always_polars), PolarsPlugin)
        assert_type(manager().plugin(always_pyarrow), ArrowPlugin)

        # Until `mypy` supports PEP 675 it won't report anything
        # https://github.com/python/mypy/issues/12554
        manager().plugin(too_dynamic)  # pyright: ignore[reportArgumentType, reportCallIssue]

        if MYPY:
            ...
        else:
            # These raise at runtime, but require a lot of extra typing to reject *everywhere*
            assert_type(manager().plugin(not_yet_1), BuiltinAny | PluginAny)
            assert_type(manager().plugin(not_yet_2), BuiltinAny | PluginAny)
            assert_type(manager().plugin(not_yet_3), BuiltinAny | PluginAny)

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

        plugin = manager().plugin("polars")

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
            if (
                not manager()
                .plugin(Implementation.PYARROW)
                .is_native_dataframe(who_knows)
            ):
                assert_type(who_knows, pd.DataFrame)
            else:
                assert_type(who_knows, pa.Table)
        else:
            assert_type(who_knows, pl.DataFrame)

    # TODO @dangotbanned: Would be nice for this to preserve the implementation
    def typing_plugin_dogfood_implementation() -> None:
        polars_1 = manager().plugin("polars").implementation
        polars_2 = manager().plugin(polars_1)
        assert_type(polars_2, Literal[Implementation.POLARS])  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure]

        pyarrow_1 = manager().plugin("polars").implementation
        pyarrow_2 = manager().plugin(pyarrow_1)
        assert_type(pyarrow_2, Literal[Implementation.PYARROW])  # type: ignore[assert-type] # pyright: ignore[reportAssertTypeFailure]

    def typing_versioned_classes_lazy(backend: IntoPlugin) -> None:
        lazy = manager().plugin(backend)
        assert_type(lazy, PluginAny | BuiltinAny)

        classes = lazy.__narwhals_classes__
        assert_type(classes, PolarsClasses | ArrowClasses | Any)
        if cc.can_lazy(classes):
            evaluator = classes.evaluator

            if MYPY:
                assert_type(evaluator, type[Any | pl_main.PlanEvaluator])
            else:
                assert_type(evaluator, type[pl_main.PlanEvaluator])

            _v1 = classes.v1  # type: ignore[union-attr]

            if cc.can_v1(classes):
                assert_type(classes.v1.lazyframe, type[pl_v1.LazyFrame])

    def typing_versioned_classes_eager(backend: IntoPlugin) -> None:
        eager = manager().plugin(backend)
        assert_type(eager, PluginAny | BuiltinAny)

        classes = eager.__narwhals_classes__
        assert_type(classes, PolarsClasses | ArrowClasses | Any)
        if cc.can_eager(classes):
            _evaluator = classes.evaluator  # type: ignore[union-attr]
            dataframe = classes.dataframe

            if MYPY:
                assert_type(dataframe, type[Any | pl_main.DataFrame | pa_main.DataFrame])
            else:
                assert_type(dataframe, type[pl_main.DataFrame | pa_main.DataFrame])

            _v2 = classes.v2  # type: ignore[union-attr]

            if cc.can_v2(classes):
                assert_type(classes.v2.series, type[pl_v2.Series | pa_v2.Series])

            if cc.can_v1(classes):
                assert_type(classes.v1.series, type[pl_v1.Series | pa_v1.Series])

    def typing_concrete_assignable_to_generic() -> None:
        """A little inference workout to check the types from one api play nicely with the other."""
        version = Version.MAIN

        # Round 1: Start with `Compliant*[NativeT]`
        dataframe = manager().dataframe("polars", version)
        df_compliant = dataframe.from_dict({})
        df_native = df_compliant.native
        df_narwhals = df_compliant.to_narwhals()
        df_narwhals = df_narwhals.from_native(df_native)
        ser_narwhals = df_narwhals.to_series()
        ser_native = ser_narwhals.to_native()
        df_native = df_native.with_columns(ser_native)
        evaluator = manager().evaluator("polars", version)

        # Round 2: Now the concrete types `Polars*`
        # Nothing before was annotated, and reassigning with a narrower (compatible) type should be okay
        classes = manager().plugin("polars").__narwhals_classes__
        dataframe = classes.dataframe
        df_compliant = dataframe.from_dict({})
        df_native = df_compliant.native
        df_narwhals = df_compliant.to_narwhals()
        df_narwhals = df_narwhals.from_native(df_native)
        ser_narwhals = df_narwhals.to_series()
        ser_native = ser_narwhals.to_native()
        df_native = df_native.with_columns(ser_native)
        evaluator = classes.evaluator

        # Round 3: Back to roughly the same as "Round 1", but using `Implementation`
        # The types should again be compatible
        dataframe = manager().dataframe(Implementation.POLARS, version)
        df_compliant = dataframe.from_dict({})
        df_native = df_compliant.native
        df_narwhals = df_compliant.to_narwhals()
        df_narwhals = df_narwhals.from_native(df_native)
        ser_narwhals = df_narwhals.to_series()
        ser_native = ser_narwhals.to_native()
        df_native = df_native.with_columns(ser_native)
        evaluator = manager().evaluator(Implementation.POLARS, version)

        # aaaaaaaaaaand did we do it?
        assert_type(df_native, pl.DataFrame)
        assert_type(df_narwhals, nwp.DataFrame[pl.DataFrame, pl.Series])
        assert_type(dataframe, type[CompliantDataFrame[pl.DataFrame, pl.Series]])
        assert_type(df_compliant, CompliantDataFrame[pl.DataFrame, pl.Series])
        assert_type(ser_native, pl.Series)
        assert_type(ser_narwhals, nwp.Series[pl.Series])
        assert_type(evaluator, type[ct.PlanEvaluator[pl.LazyFrame]])

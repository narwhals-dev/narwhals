from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from narwhals._plan.compliant.plugins import PluginAny, load_plugin
from narwhals._typing_compat import assert_never
from tests.plan.utils import re_compile


def test_load_plugin() -> None:
    polars_plugin = load_plugin("polars")
    pyarrow_plugin = load_plugin("pyarrow")
    assert polars_plugin.plugin_name == "polars"
    assert pyarrow_plugin.plugin_name == "pyarrow"

    assert hasattr(polars_plugin, "implementation")
    assert polars_plugin.implementation.is_polars()

    assert hasattr(pyarrow_plugin, "implementation")
    assert pyarrow_plugin.implementation.is_pyarrow()

    with pytest.raises(NotImplementedError, match=r"not yet"):
        assert_never(load_plugin("modin"))

    with pytest.raises(
        TypeError, match=re_compile(r"Unsupported `backend` .+got: 'i dont exist'")
    ):
        load_plugin("i dont exist")


XFAIL_TODO = pytest.mark.xfail(reason="TODO", raises=NotImplementedError)


# TODO @dangotbanned: Cover when connected up to the rest
@XFAIL_TODO
def test_plugin_is_loaded() -> None:  # pragma: no cover
    raise NotImplementedError


# TODO @dangotbanned: Cover when connected up to the rest
@XFAIL_TODO
def test_plugin_is_available() -> None:  # pragma: no cover
    raise NotImplementedError


if TYPE_CHECKING:
    import random
    from types import ModuleType

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import LiteralString, TypeAlias, assert_type

    from narwhals._plan.arrow import ArrowPlugin
    from narwhals._plan.polars import PolarsPlugin
    from narwhals._typing import (
        Arrow,
        Backend,
        EagerAllowed,
        IntoBackend,
        LazyAllowed,
        Polars,
    )
    from narwhals._utils import Implementation

    SupportedBuiltin: TypeAlias = Literal[
        "polars", "pyarrow", Implementation.POLARS, Implementation.PYARROW
    ]

    def typing_load_plugin(
        never_builtin: LiteralString,
        always_builtin_1: ModuleType,
        always_builtin_2: SupportedBuiltin,
        always_polars: Polars,
        always_pyarrow: Arrow,
        too_dynamic: str,
        not_yet_1: IntoBackend[Backend],
        not_yet_2: EagerAllowed,
        not_yet_3: LazyAllowed,
    ) -> None:
        # NOTE: Purely checking the result of matching `@overload`s
        assert_type(load_plugin(never_builtin), PluginAny)
        assert_type(load_plugin(always_builtin_1), ArrowPlugin | PolarsPlugin)
        assert_type(load_plugin(always_builtin_2), ArrowPlugin | PolarsPlugin)
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

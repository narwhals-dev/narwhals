from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from typing_extensions import LiteralString

    from narwhals.typing import EagerAllowed, LazyAllowed
    from tests.plan.utils import (
        ConstructorFixtureName,
        DataFrame,
        Identifier,
        LazyFrame,
        Series,
        TestBackendAny,
    )

    @pytest.fixture
    def lazyframe() -> LazyFrame: ...
    @pytest.fixture
    def dataframe() -> DataFrame: ...
    @pytest.fixture
    def series() -> Series: ...
    @pytest.fixture
    def eager() -> EagerAllowed: ...
    @pytest.fixture
    def eager_or_false() -> EagerAllowed | Literal[False]: ...
    @pytest.fixture
    def lazy() -> LazyAllowed: ...


_MAIN_DEFAULT_CONSTRUCTORS = (
    "pandas,pandas[pyarrow],polars[eager],pyarrow,duckdb,sqlframe,ibis"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--plan-include",
        default="ALL",
        help="Backend(s) that should be selected if available.",
    )
    parser.addoption(
        "--plan-exclude",
        default="",
        help="Backend(s) that should not be selected, has lower precedence than `--include`",
    )


_option = cast("Callable[[pytest.Config, str], Any]", pytest.Config.getoption)
"""Hack to avoid `pyright` using `Any | None` as the return type."""


def _resolve_options(
    config: pytest.Config,
) -> tuple[Collection[Identifier] | Literal["ALL"], Collection[Identifier] | None]:
    """Convert command line options into `include` and `exclude` sets.

    Piggybacks off of `--constructors`, if that was used instead of `--plan-include` or `--plan-exclude`.
    """
    include: set[Identifier] = set()
    exclude: set[Identifier] = set()

    # Try and integrate a lil bit
    opt_main: str = _option(config, "--constructors")
    if opt_main != _MAIN_DEFAULT_CONSTRUCTORS:
        relevant = {"pyarrow", "polars[eager]", "polars[lazy]"}
        constructors = relevant.intersection(opt_main.split(","))
        if "pyarrow" not in constructors:
            exclude.add("pyarrow")
        if constructors.isdisjoint({"polars[eager]", "polars[lazy]"}):
            exclude.add("polars")

    # The actual options specific to `narwhals._plan`
    opt_include: LiteralString = _option(config, "--plan-include")
    if opt_include != "ALL":
        include.update(opt_include.split(","))
    opt_exclude: LiteralString = _option(config, "--plan-exclude")
    if opt_exclude != "":
        exclude.update(opt_exclude.split(","))

    return (include or "ALL"), (exclude or None)


get_identifier: Callable[[Any], str] = attrgetter("identifier")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    from tests.plan.utils import TestBackend

    include, exclude = _resolve_options(metafunc.config)
    test_backends = TestBackend.prepare_backends(include=include, exclude=exclude)
    for constructor in "lazyframe", "dataframe", "series":
        _parametrize_constructor_fixture(constructor, metafunc, test_backends)

    if {"eager", "eager_or_false"}.intersection(metafunc.fixturenames):
        eager_values = []
        eager_ids = []
        for backend in test_backends:
            if eager := getattr(backend, "backend_eager", None):
                eager_values.append(eager)
                eager_ids.append(backend.identifier)
        if eager_values:
            if "eager" in metafunc.fixturenames:
                metafunc.parametrize("eager", eager_values, ids=eager_ids)
            if "eager_or_false" in metafunc.fixturenames:
                metafunc.parametrize(
                    "eager_or_false", (*eager_values, False), ids=(*eager_ids, "False")
                )
    if "lazy" in metafunc.fixturenames and (
        lazy_values_ids := [
            (lazy, backend.identifier)
            for backend in test_backends
            if (lazy := getattr(backend, "backend_lazy", None))
        ]
    ):
        lazy_values, lazy_ids = zip(*lazy_values_ids)
        metafunc.parametrize("lazy", lazy_values, ids=lazy_ids)


def _parametrize_constructor_fixture(
    name: ConstructorFixtureName,
    metafunc: pytest.Metafunc,
    backends: tuple[TestBackendAny, ...],
) -> None:
    if name in metafunc.fixturenames and (
        impls := [
            impl for backend in backends if (impl := backend.try_get_constructor(name))
        ]
    ):
        metafunc.parametrize(name, impls, ids=get_identifier)

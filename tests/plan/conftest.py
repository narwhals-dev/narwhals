from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Collection

    import pytest
    from typing_extensions import LiteralString

    from narwhals.typing import EagerAllowed
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


def _resolve_options(
    config: pytest.Config,
) -> tuple[Collection[Identifier] | Literal["ALL"], Collection[Identifier] | None]:
    """Convert command line options into `include` and `exclude` sets.

    Piggybacks off of `--constructors`, if that was used instead of `--plan-include` or `--plan-exclude`.
    """
    include: set[Identifier] = set()
    exclude: set[Identifier] = set()

    # Try and integrate a lil bit
    opt_main: str = config.getoption("--constructors")
    if opt_main != _MAIN_DEFAULT_CONSTRUCTORS:
        relevant = {"pyarrow", "polars[eager]", "polars[lazy]"}
        constructors = relevant.intersection(opt_main.split(","))
        if "pyarrow" not in constructors:
            exclude.add("pyarrow")
        if constructors.isdisjoint({"polars[eager]", "polars[lazy]"}):
            exclude.add("polars")

    # The actual options specific to `narwhals._plan`
    opt_include: LiteralString = config.getoption("--plan-include")
    if opt_include != "ALL":
        include.update(opt_include.split(","))
    opt_exclude: LiteralString = config.getoption("--plan-exclude")
    if opt_exclude != "":
        exclude.update(opt_exclude.split(","))

    return (include or "ALL"), (exclude or None)


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


def _parametrize_constructor_fixture(
    name: ConstructorFixtureName,
    metafunc: pytest.Metafunc,
    backends: tuple[TestBackendAny, ...],
) -> None:
    if name in metafunc.fixturenames:
        methods = []
        ids = []
        for backend in backends:
            if constructor := backend.try_get_constructor(name):
                methods.append(constructor)
                # TODO @dangotbanned: Replace this `ids` bit with a callable (e.g. `attrgetter("identifier")`)
                # Need to finish converting the bound methods to instances *first though
                ids.append(backend.identifier)
        if methods:
            metafunc.parametrize(name, methods, ids=ids)

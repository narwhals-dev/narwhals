from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

import narwhals as nw
from tests.conftest import TEST_EAGER_BACKENDS

if TYPE_CHECKING:
    from collections.abc import Collection

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


_HAS_IMPLEMENTATION = frozenset((nw.Implementation.PYARROW, "pyarrow"))
"""Using to filter *the source* of `eager_backend` - which includes `polars` and `pandas` when available.

For now, this lets some tests be written in a backend agnostic way.
"""


@pytest.fixture(
    scope="session", params=_HAS_IMPLEMENTATION.intersection(TEST_EAGER_BACKENDS)
)
def eager(request: pytest.FixtureRequest) -> EagerAllowed:
    result: EagerAllowed = request.param
    return result


_HAS_IMPLEMENTATION_IMPL = frozenset(
    el for el in _HAS_IMPLEMENTATION if isinstance(el, nw.Implementation)
)
"""Filtered for heavily parametric tests."""


@pytest.fixture(
    scope="session",
    params=_HAS_IMPLEMENTATION_IMPL.intersection(TEST_EAGER_BACKENDS).union([False]),
)
def eager_or_false(request: pytest.FixtureRequest) -> EagerAllowed | Literal[False]:
    result: EagerAllowed | Literal[False] = request.param
    return result


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
    for name in "lazyframe", "dataframe", "series":
        _parametrize_constructor_fixture(name, metafunc, test_backends)


def _parametrize_constructor_fixture(
    name: ConstructorFixtureName,
    metafunc: pytest.Metafunc,
    backends: tuple[TestBackendAny, ...],
) -> None:
    if name in metafunc.fixturenames:
        methods = []
        ids = []
        for backend in backends:
            if backend.supports[name]:
                methods.append(getattr(backend, name))
                ids.append(backend.identifier)
        if methods:
            metafunc.parametrize(name, methods, ids=ids)

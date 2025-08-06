from __future__ import annotations

import os
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, cast

import pytest

from narwhals._utils import Implementation
from narwhals.testing._utils import (
    EAGER_CONSTRUCTORS,
    GPU_CONSTRUCTORS,
    LAZY_CONSTRUCTORS,
    get_constructors,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from narwhals._namespace import EagerAllowed
    from tests.utils import Constructor

    Data: TypeAlias = "dict[str, list[Any]]"

# When testing cudf.pandas in Kaggle, we get an error if we try to run
# python -m cudf.pandas -m pytest --constructors=pandas. This gives us
# a way to run `python -m cudf.pandas -m pytest` and control which constructors
# get tested.
if default_constructors := os.environ.get(
    "NARWHALS_DEFAULT_CONSTRUCTORS", None
):  # pragma: no cover
    DEFAULT_CONSTRUCTORS = default_constructors
else:
    DEFAULT_CONSTRUCTORS = (
        "pandas,pandas[pyarrow],polars[eager],pyarrow,duckdb,sqlframe,ibis"
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--all-cpu-constructors",
        action="store_true",
        default=False,
        help="run tests with all cpu constructors",
    )
    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="libraries to test",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Function]
) -> None:  # pragma: no cover
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("all_cpu_constructors"):  # pragma: no cover
        selected_constructors: list[str] = [
            *iter(EAGER_CONSTRUCTORS.keys()),
            *iter(LAZY_CONSTRUCTORS.keys()),
        ]
        selected_constructors = [
            x
            for x in selected_constructors
            if x not in GPU_CONSTRUCTORS
            and x
            not in {
                "modin",  # too slow
                "spark[connect]",  # complex local setup; can't run together with local spark
            }
        ]
    else:  # pragma: no cover
        opt = cast("str", metafunc.config.getoption("constructors"))
        selected_constructors = opt.split(",")

    (
        eager_constructors,
        eager_constructors_ids,
        lazy_constructors,
        lazy_constructors_ids,
    ) = get_constructors(selected_constructors=selected_constructors)

    constructors: list[Constructor] = [*eager_constructors, *lazy_constructors]
    constructors_ids: list[str] = [*eager_constructors_ids, *lazy_constructors_ids]

    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize(
            "constructor_eager", eager_constructors, ids=eager_constructors_ids
        )
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize("constructor", constructors, ids=constructors_ids)


TEST_EAGER_BACKENDS: list[EagerAllowed] = []
TEST_EAGER_BACKENDS.extend(
    (Implementation.POLARS, "polars") if find_spec("polars") is not None else ()
)
TEST_EAGER_BACKENDS.extend(
    (Implementation.PANDAS, "pandas") if find_spec("pandas") is not None else ()
)
TEST_EAGER_BACKENDS.extend(
    (Implementation.PYARROW, "pyarrow") if find_spec("pyarrow") is not None else ()
)


@pytest.fixture(params=TEST_EAGER_BACKENDS)
def eager_backend(request: pytest.FixtureRequest) -> EagerAllowed:
    return request.param  # type: ignore[no-any-return]

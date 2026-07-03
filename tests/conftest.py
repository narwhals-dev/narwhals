from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import Implementation

# `narwhals.testing.pytest_plugin` registers itself via the `pytest11` entry point (see pyproject.toml)
# so it auto-loads as soon as Narwhals is installed.
# That plugin is what owns the `--nw-backends`, `--all-nw-backends`, and `--use-external-constructor`
# CLI options as well as parametrising the `constructor*` fixtures.

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._typing import EagerAllowed
    from narwhals.testing.typing import DataFrameConstructor, FrameConstructor
    from narwhals.typing import NonNestedDType
    from tests.utils import NestedOrEnumDType


# Narwhals-internal pytest options (not part of the public testing plugin)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Function]
) -> None:  # pragma: no cover
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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


@pytest.fixture(params=[el for el in TEST_EAGER_BACKENDS if not isinstance(el, str)])
def eager_implementation(request: pytest.FixtureRequest) -> EagerAllowed:
    """Use if a test is heavily parametric, skips `str` backend."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        nw.Boolean,
        nw.Categorical,
        nw.Date,
        nw.Datetime,
        nw.Decimal,
        nw.Duration,
        nw.Float16,
        nw.Float32,
        nw.Float64,
        nw.Int8,
        nw.Int16,
        nw.Int32,
        nw.Int64,
        nw.Int128,
        nw.Object,
        nw.String,
        nw.Time,
        nw.UInt8,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
        nw.UInt128,
        nw.Unknown,
        nw.Binary,
    ],
    ids=lambda tp: tp.__name__,
)
def non_nested_type(request: pytest.FixtureRequest) -> type[NonNestedDType]:
    tp_dtype: type[NonNestedDType] = request.param
    return tp_dtype


@pytest.fixture(
    params=[
        nw.List(nw.Float32),
        nw.Array(nw.String, 2),
        nw.Struct({"a": nw.Boolean}),
        nw.Enum(["beluga", "narwhal"]),
    ],
    ids=lambda obj: type(obj).__name__,
)
def nested_dtype(request: pytest.FixtureRequest) -> NestedOrEnumDType:
    dtype: NestedOrEnumDType = request.param
    return dtype


# The following fixtures are short-name aliases of those registered in
# `narwhals/testing/pytest_plugin.py`. Calling a constructor without an explicit
# `namespace` defaults to the main `narwhals` namespace; tests can still pass
# `nw_v1` / `nw_v2` explicitly to opt in to a stable namespace. The legacy pattern
# `nw.from_native(constructor(data))` keeps working because `nw.from_native` is
# idempotent on narwhals objects.
# TODO(FBruzzesi): Drop these aliases once every test requests `nw_frame` /
# `nw_dataframe` / `nw_pandas_like_frame` directly.


@pytest.fixture
def constructor(nw_frame: FrameConstructor) -> FrameConstructor:
    return nw_frame


@pytest.fixture
def constructor_eager(nw_dataframe: DataFrameConstructor) -> DataFrameConstructor:
    return nw_dataframe


@pytest.fixture
def constructor_pandas_like(
    nw_pandas_like_frame: DataFrameConstructor,
) -> DataFrameConstructor:
    return nw_pandas_like_frame

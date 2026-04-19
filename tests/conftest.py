from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation

# `narwhals.testing.pytest_plugin` registers itself via the `pytest11` entry point (see pyproject.toml)
# so it auto-loads as soon as Narwhals is installed.
# That plugin is what owns the `--constructors`, `--all-cpu-constructors`, and `--use-external-constructor`
# CLI options as well as parametrising the `constructor*` fixtures.

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from narwhals._typing import EagerAllowed
    from narwhals.testing.typing import EagerFrameConstructor, FrameConstructor
    from narwhals.typing import NonNestedDType
    from tests.utils import NestedOrEnumDType

    Data: TypeAlias = "dict[str, list[Any]]"


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


# The following fixtures are aliases of those registered in `narwhals/testing/pytest_plugin.py`
# in order to be backward compatible with the old fixture names and avoid having to change
# every single test.
# TODO(FBruzzesi): Rm once all tests start using nw_frame_constructor directly
@pytest.fixture
def constructor(nw_frame_constructor: FrameConstructor) -> FrameConstructor:
    return nw_frame_constructor


@pytest.fixture
def constructor_eager(nw_eager_constructor: EagerFrameConstructor) -> FrameConstructor:
    return nw_eager_constructor


@pytest.fixture
def constructor_pandas_like(
    nw_pandas_like_constructor: EagerFrameConstructor,
) -> FrameConstructor:
    return nw_pandas_like_constructor

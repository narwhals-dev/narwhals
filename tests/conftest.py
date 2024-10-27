from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_dask_dataframe
from narwhals.dependencies import get_modin
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from narwhals.typing import IntoFrame
    from tests.utils import Constructor

with contextlib.suppress(ImportError):
    import modin.pandas  # noqa: F401
with contextlib.suppress(ImportError):
    import dask.dataframe  # noqa: F401
with contextlib.suppress(ImportError):
    import cudf  # noqa: F401


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Any, items: Any) -> Any:  # pragma: no cover
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pandas_constructor(obj: Any) -> IntoDataFrame:
    return pd.DataFrame(obj)  # type: ignore[no-any-return]


def pandas_nullable_constructor(obj: Any) -> IntoDataFrame:
    return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")  # type: ignore[no-any-return]


def pandas_pyarrow_constructor(obj: Any) -> IntoDataFrame:
    return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def modin_constructor(obj: Any) -> IntoDataFrame:  # pragma: no cover
    mpd = get_modin()
    return mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def cudf_constructor(obj: Any) -> IntoDataFrame:  # pragma: no cover
    cudf = get_cudf()
    return cudf.DataFrame(obj)  # type: ignore[no-any-return]


def polars_eager_constructor(obj: Any) -> IntoDataFrame:
    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: Any) -> pl.LazyFrame:
    return pl.LazyFrame(obj)


def dask_lazy_p1_constructor(obj: Any) -> IntoFrame:  # pragma: no cover
    dd = get_dask_dataframe()
    return dd.from_dict(obj, npartitions=1)  # type: ignore[no-any-return]


def dask_lazy_p2_constructor(obj: Any) -> IntoFrame:  # pragma: no cover
    dd = get_dask_dataframe()
    return dd.from_dict(obj, npartitions=2)  # type: ignore[no-any-return]


def pyarrow_table_constructor(obj: Any) -> IntoDataFrame:
    return pa.table(obj)  # type: ignore[no-any-return]


if PANDAS_VERSION >= (2, 0, 0):
    eager_constructors = [
        pandas_constructor,
        pandas_nullable_constructor,
        pandas_pyarrow_constructor,
    ]
else:  # pragma: no cover
    eager_constructors = [pandas_constructor]

eager_constructors.extend([polars_eager_constructor, pyarrow_table_constructor])
lazy_constructors = [polars_lazy_constructor]

if get_modin() is not None:  # pragma: no cover
    eager_constructors.append(modin_constructor)
if get_cudf() is not None:
    eager_constructors.append(cudf_constructor)  # pragma: no cover
if get_dask_dataframe() is not None:  # pragma: no cover
    # TODO(unassigned): reinstate both dask constructors once if/when we have a dask use-case
    # lazy_constructors.extend([dask_lazy_p1_constructor, dask_lazy_p2_constructor])  # noqa: ERA001
    lazy_constructors.append(dask_lazy_p2_constructor)  # type: ignore  # noqa: PGH003


@pytest.fixture(params=eager_constructors)
def constructor_eager(
    request: pytest.FixtureRequest,
) -> Callable[[Any], IntoDataFrame]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[*eager_constructors, *lazy_constructors])
def constructor(request: pytest.FixtureRequest) -> Constructor:
    return request.param  # type: ignore[no-any-return]

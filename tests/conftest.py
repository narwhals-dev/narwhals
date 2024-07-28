from typing import Any
from typing import Callable

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from narwhals.dependencies import get_dask_dataframe
from narwhals.dependencies import get_modin
from narwhals.typing import IntoDataFrame
from narwhals.typing import IntoFrame
from narwhals.utils import parse_version


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
    return mpd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def polars_eager_constructor(obj: Any) -> IntoDataFrame:
    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: Any) -> pl.LazyFrame:
    return pl.LazyFrame(obj)


def dask_lazy_constructor(obj: Any) -> IntoFrame:
    dd = get_dask_dataframe()
    return dd.from_pandas(pd.DataFrame(obj))  # type: ignore[no-any-return]


def pyarrow_table_constructor(obj: Any) -> IntoDataFrame:
    return pa.table(obj)  # type: ignore[no-any-return]


if parse_version(pd.__version__) >= parse_version("2.0.0"):
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
if get_dask_dataframe() is not None:  # pragma: no cover
    lazy_constructors.append(dask_lazy_constructor)  # type: ignore[arg-type]


@pytest.fixture(params=eager_constructors)
def constructor(request: Any) -> Callable[[Any], IntoDataFrame]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[*eager_constructors, *lazy_constructors])
def constructor_lazy(request: Any) -> Callable[[Any], Any]:
    return request.param  # type: ignore[no-any-return]

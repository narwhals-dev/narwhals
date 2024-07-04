from typing import Any
from typing import Callable

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from narwhals.dependencies import get_modin
from narwhals.typing import IntoDataFrame
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
    return pd.DataFrame(obj).convert_dtypes()  # type: ignore[no-any-return]


def pandas_pyarrow_constructor(obj: Any) -> IntoDataFrame:
    return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def modin_constructor(obj: Any) -> IntoDataFrame:  # pragma: no cover
    mpd = get_modin()
    return mpd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def polars_constructor(obj: Any) -> IntoDataFrame:
    return pl.DataFrame(obj)


if parse_version(pd.__version__) >= parse_version("2.0.0"):
    params = [pandas_constructor, pandas_nullable_constructor, pandas_pyarrow_constructor]
else:  # pragma: no cover
    params = [pandas_constructor]
params.append(polars_constructor)
if get_modin() is not None:  # pragma: no cover
    params.append(modin_constructor)


@pytest.fixture(params=params)
def constructor(request: Any) -> Callable[[Any], IntoDataFrame]:
    return request.param  # type: ignore[no-any-return]


# TODO: once pyarrow has complete coverage, we can remove this one,
# and just put `pa.table` into `constructor`
@pytest.fixture(params=[*params, pa.table])
def constructor_with_pyarrow(request: Any) -> Callable[[Any], IntoDataFrame]:
    return request.param  # type: ignore[no-any-return]


def pandas_series_constructor(obj: Any) -> Any:
    return pd.Series(obj)


def pandas_series_nullable_constructor(obj: Any) -> Any:
    return pd.Series(obj).convert_dtypes()


def pandas_series_pyarrow_constructor(obj: Any) -> Any:
    return pd.Series(obj).convert_dtypes(dtype_backend="pyarrow")


def modin_series_constructor(obj: Any) -> Any:  # pragma: no cover
    mpd = get_modin()
    return mpd.Series(obj).convert_dtypes(dtype_backend="pyarrow")


def polars_series_constructor(obj: Any) -> Any:
    return pl.Series(obj)


if parse_version(pd.__version__) >= parse_version("2.0.0"):
    params_series = [
        pandas_series_constructor,
        pandas_series_nullable_constructor,
        pandas_series_pyarrow_constructor,
    ]
else:  # pragma: no cover
    params_series = [pandas_series_constructor]
params_series.append(polars_series_constructor)
if get_modin() is not None:  # pragma: no cover
    params_series.append(modin_series_constructor)


@pytest.fixture(params=params_series)
def constructor_series(request: Any) -> Callable[[Any], Any]:
    return request.param  # type: ignore[no-any-return]


# TODO: once pyarrow has complete coverage, we can remove this one,
# and just put `pa.table` into `constructor`
@pytest.fixture(params=[*params_series, lambda x: pa.chunked_array([x])])
def constructor_series_with_pyarrow(request: Any) -> Callable[[Any], Any]:
    return request.param  # type: ignore[no-any-return]

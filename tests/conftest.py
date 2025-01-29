from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from narwhals.utils import generate_temporary_column_name
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    import duckdb

    from narwhals.typing import IntoDataFrame
    from narwhals.typing import IntoFrame

MIN_PANDAS_NULLABLE_VERSION = (1, 5)

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
        "pandas,pandas[nullable],pandas[pyarrow],polars[eager],polars[lazy],pyarrow"
    )


def pytest_addoption(parser: Any) -> None:
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


def pandas_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:
    return pd.DataFrame(obj)  # type: ignore[no-any-return]


def pandas_nullable_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:
    return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")  # type: ignore[no-any-return]


def pandas_pyarrow_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:
    return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def modin_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:  # pragma: no cover
    import modin.pandas as mpd

    return mpd.DataFrame(pd.DataFrame(obj))  # type: ignore[no-any-return]


def modin_pyarrow_constructor(
    obj: dict[str, list[Any]],
) -> IntoDataFrame:  # pragma: no cover
    import modin.pandas as mpd

    return mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def cudf_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:  # pragma: no cover
    import cudf

    return cudf.DataFrame(obj)  # type: ignore[no-any-return]


def polars_eager_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:
    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: dict[str, list[Any]]) -> pl.LazyFrame:
    return pl.LazyFrame(obj)


def duckdb_lazy_constructor(obj: dict[str, list[Any]]) -> duckdb.DuckDBPyRelation:
    import duckdb

    _df = pl.LazyFrame(obj)
    return duckdb.table("_df")


def dask_lazy_p1_constructor(obj: dict[str, list[Any]]) -> IntoFrame:  # pragma: no cover
    import dask.dataframe as dd

    return dd.from_dict(obj, npartitions=1)  # type: ignore[no-any-return]


def dask_lazy_p2_constructor(obj: dict[str, list[Any]]) -> IntoFrame:  # pragma: no cover
    import dask.dataframe as dd

    return dd.from_dict(obj, npartitions=2)  # type: ignore[no-any-return]


def pyarrow_table_constructor(obj: dict[str, list[Any]]) -> IntoDataFrame:
    return pa.table(obj)  # type: ignore[no-any-return]


def pyspark_lazy_constructor() -> Callable[[Any], IntoFrame]:  # pragma: no cover
    try:
        from pyspark.sql import SparkSession
    except ImportError:  # pragma: no cover
        pytest.skip("pyspark is not installed")
        return None

    import warnings
    from atexit import register

    with warnings.catch_warnings():
        # The spark session seems to trigger a polars warning.
        # Polars is imported in the tests, but not used in the spark operations
        warnings.filterwarnings(
            "ignore", r"Using fork\(\) can cause Polars", category=RuntimeWarning
        )

        session = (
            SparkSession.builder.appName("unit-tests")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            # executing one task at a time makes the tests faster
            .config("spark.default.parallelism", "1")
            .config("spark.sql.shuffle.partitions", "2")
            # common timezone for all tests environments
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

        register(session.stop)

        def _constructor(obj: dict[str, list[Any]]) -> IntoFrame:
            _obj = deepcopy(obj)
            index_col_name = generate_temporary_column_name(n_bytes=8, columns=list(_obj))
            _obj[index_col_name] = list(range(len(_obj[next(iter(_obj))])))

            return (  # type: ignore[no-any-return]
                session.createDataFrame([*zip(*_obj.values())], schema=[*_obj.keys()])
                .repartition(2)
                .orderBy(index_col_name)
                .drop(index_col_name)
            )

        return _constructor


def sqlframe_pyspark_lazy_constructor(
    obj: dict[str, Any],
) -> Callable[[Any], IntoFrame]:  # pragma: no cover
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    return (  # type: ignore[no-any-return]
        session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()])
    )


EAGER_CONSTRUCTORS: dict[str, Callable[[Any], IntoDataFrame]] = {
    "pandas": pandas_constructor,
    "pandas[nullable]": pandas_nullable_constructor,
    "pandas[pyarrow]": pandas_pyarrow_constructor,
    "pyarrow": pyarrow_table_constructor,
    "modin": modin_constructor,
    "modin[pyarrow]": modin_pyarrow_constructor,
    "cudf": cudf_constructor,
    "polars[eager]": polars_eager_constructor,
}
LAZY_CONSTRUCTORS: dict[str, Callable[[Any], IntoFrame]] = {
    "dask": dask_lazy_p2_constructor,
    "polars[lazy]": polars_lazy_constructor,
    "duckdb": duckdb_lazy_constructor,
    "pyspark": pyspark_lazy_constructor,  # type: ignore[dict-item]
    # We've reported several bugs to sqlframe - once they address
    # them, we can start testing them as part of our CI.
    # "sqlframe": pyspark_lazy_constructor,  # noqa: ERA001
}
GPU_CONSTRUCTORS: dict[str, Callable[[Any], IntoFrame]] = {"cudf": cudf_constructor}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("all_cpu_constructors"):
        selected_constructors: list[str] = [
            *iter(EAGER_CONSTRUCTORS.keys()),
            *iter(LAZY_CONSTRUCTORS.keys()),
        ]
        selected_constructors = [
            x
            for x in selected_constructors
            if x not in GPU_CONSTRUCTORS and x not in "modin"  # too slow
        ]
    else:  # pragma: no cover
        selected_constructors = metafunc.config.getoption("constructors").split(",")

    eager_constructors: list[Callable[[Any], IntoDataFrame]] = []
    eager_constructors_ids: list[str] = []
    constructors: list[Callable[[Any], IntoFrame]] = []
    constructors_ids: list[str] = []

    for constructor in selected_constructors:
        if (
            constructor in ("pandas[nullable]", "pandas[pyarrow]")
            and MIN_PANDAS_NULLABLE_VERSION > PANDAS_VERSION
        ):  # pragma: no cover
            continue
        if constructor in EAGER_CONSTRUCTORS:
            eager_constructors.append(EAGER_CONSTRUCTORS[constructor])
            eager_constructors_ids.append(constructor)
            constructors.append(EAGER_CONSTRUCTORS[constructor])
            constructors_ids.append(constructor)
        elif constructor in LAZY_CONSTRUCTORS:
            if constructor == "pyspark":
                constructors.append(pyspark_lazy_constructor())
            else:
                constructors.append(LAZY_CONSTRUCTORS[constructor])
            constructors_ids.append(constructor)
        else:  # pragma: no cover
            msg = f"Expected one of {EAGER_CONSTRUCTORS.keys()} or {LAZY_CONSTRUCTORS.keys()}, got {constructor}"
            raise ValueError(msg)

    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize(
            "constructor_eager", eager_constructors, ids=eager_constructors_ids
        )
    elif "constructor" in metafunc.fixturenames:
        if (
            any(x in str(metafunc.module) for x in ("unpivot", "from_dict", "from_numpy"))
            and LAZY_CONSTRUCTORS["duckdb"] in constructors
        ):
            constructors.remove(LAZY_CONSTRUCTORS["duckdb"])
            constructors_ids.remove("duckdb")
        metafunc.parametrize("constructor", constructors, ids=constructors_ids)

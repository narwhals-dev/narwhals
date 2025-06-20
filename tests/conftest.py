from __future__ import annotations

import os
import uuid
from copy import deepcopy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, cast

import pytest

from narwhals._utils import generate_temporary_column_name
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb
    import ibis
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from ibis.backends.duckdb import Backend as IbisDuckDBBackend
    from pyspark.sql import DataFrame as PySparkDataFrame
    from typing_extensions import TypeAlias

    from narwhals._spark_like.dataframe import SQLFrameDataFrame
    from narwhals.typing import NativeFrame, NativeLazyFrame
    from tests.utils import Constructor, ConstructorEager

    Data: TypeAlias = "dict[str, list[Any]]"

MIN_PANDAS_NULLABLE_VERSION = (2,)

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


def pandas_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj)


def pandas_nullable_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")


def pandas_pyarrow_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")


def modin_constructor(obj: Data) -> NativeFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj))
    return cast("NativeFrame", df)


def modin_pyarrow_constructor(obj: Data) -> NativeFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")
    return cast("NativeFrame", df)


def cudf_constructor(obj: Data) -> NativeFrame:  # pragma: no cover
    import cudf

    df = cudf.DataFrame(obj)
    return cast("NativeFrame", df)


def polars_eager_constructor(obj: Data) -> pl.DataFrame:
    import polars as pl

    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: Data) -> pl.LazyFrame:
    import polars as pl

    return pl.LazyFrame(obj)


def duckdb_lazy_constructor(obj: Data) -> duckdb.DuckDBPyRelation:
    import duckdb
    import polars as pl

    duckdb.sql("""set timezone = 'UTC'""")

    _df = pl.LazyFrame(obj)
    return duckdb.table("_df")


def dask_lazy_p1_constructor(obj: Data) -> NativeLazyFrame:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeLazyFrame", dd.from_dict(obj, npartitions=1))


def dask_lazy_p2_constructor(obj: Data) -> NativeLazyFrame:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeLazyFrame", dd.from_dict(obj, npartitions=2))


def pyarrow_table_constructor(obj: dict[str, Any]) -> pa.Table:
    import pyarrow as pa

    return pa.table(obj)


def pyspark_lazy_constructor() -> Callable[[Data], PySparkDataFrame]:  # pragma: no cover
    pytest.importorskip("pyspark")
    import warnings
    from atexit import register

    is_spark_connect = bool(os.environ.get("SPARK_CONNECT", None))

    if TYPE_CHECKING:
        from pyspark.sql import SparkSession
    elif is_spark_connect:
        from pyspark.sql.connect.session import SparkSession
    else:
        from pyspark.sql import SparkSession

    with warnings.catch_warnings():
        # The spark session seems to trigger a polars warning.
        # Polars is imported in the tests, but not used in the spark operations
        warnings.filterwarnings(
            "ignore", r"Using fork\(\) can cause Polars", category=RuntimeWarning
        )
        builder = cast("SparkSession.Builder", SparkSession.builder).appName("unit-tests")

        session = (
            (
                builder.remote(f"sc://localhost:{os.environ.get('SPARK_PORT', '15002')}")
                if is_spark_connect
                else builder.master("local[1]").config("spark.ui.enabled", "false")
            )
            .config("spark.default.parallelism", "1")
            .config("spark.sql.shuffle.partitions", "2")
            # common timezone for all tests environments
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

        register(session.stop)

        def _constructor(obj: Data) -> PySparkDataFrame:
            _obj = deepcopy(obj)
            index_col_name = generate_temporary_column_name(n_bytes=8, columns=list(_obj))
            _obj[index_col_name] = list(range(len(_obj[next(iter(_obj))])))

            return (
                session.createDataFrame([*zip(*_obj.values())], schema=[*_obj.keys()])
                .repartition(2)
                .orderBy(index_col_name)
                .drop(index_col_name)
            )

        return _constructor


def sqlframe_pyspark_lazy_constructor(obj: Data) -> SQLFrameDataFrame:  # pragma: no cover
    from sqlframe.duckdb import DuckDBSession

    session = DuckDBSession()
    return session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()])


@lru_cache(maxsize=1)
def _ibis_backend() -> IbisDuckDBBackend:  # pragma: no cover
    """Cached (singleton) in-memory backend to ensure all tables exist within the same in-memory database."""
    import ibis

    return ibis.duckdb.connect()


def ibis_lazy_constructor(obj: Data) -> ibis.Table:  # pragma: no cover
    import polars as pl

    ldf = pl.from_dict(obj).lazy()
    table_name = str(uuid.uuid4())
    return _ibis_backend().create_table(table_name, ldf)


EAGER_CONSTRUCTORS: dict[str, ConstructorEager] = {
    "pandas": pandas_constructor,
    "pandas[nullable]": pandas_nullable_constructor,
    "pandas[pyarrow]": pandas_pyarrow_constructor,
    "pyarrow": pyarrow_table_constructor,
    "modin": modin_constructor,
    "modin[pyarrow]": modin_pyarrow_constructor,
    "cudf": cudf_constructor,
    "polars[eager]": polars_eager_constructor,
}
LAZY_CONSTRUCTORS: dict[str, Constructor] = {
    "dask": dask_lazy_p2_constructor,
    "polars[lazy]": polars_lazy_constructor,
    "duckdb": duckdb_lazy_constructor,
    "pyspark": pyspark_lazy_constructor,  # type: ignore[dict-item]
    "sqlframe": sqlframe_pyspark_lazy_constructor,
    "ibis": ibis_lazy_constructor,
}
GPU_CONSTRUCTORS: dict[str, ConstructorEager] = {"cudf": cudf_constructor}


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

    eager_constructors: list[ConstructorEager] = []
    eager_constructors_ids: list[str] = []
    constructors: list[Constructor] = []
    constructors_ids: list[str] = []

    for constructor in selected_constructors:
        if (
            constructor in {"pandas[nullable]", "pandas[pyarrow]"}
            and MIN_PANDAS_NULLABLE_VERSION > PANDAS_VERSION
        ):
            continue  # pragma: no cover

        if constructor in EAGER_CONSTRUCTORS:
            eager_constructors.append(EAGER_CONSTRUCTORS[constructor])
            eager_constructors_ids.append(constructor)
            constructors.append(EAGER_CONSTRUCTORS[constructor])
        elif constructor in {"pyspark", "pyspark[connect]"}:  # pragma: no cover
            constructors.append(pyspark_lazy_constructor())
        elif constructor in LAZY_CONSTRUCTORS:
            constructors.append(LAZY_CONSTRUCTORS[constructor])
        else:  # pragma: no cover
            msg = f"Expected one of {EAGER_CONSTRUCTORS.keys()} or {LAZY_CONSTRUCTORS.keys()}, got {constructor}"
            raise ValueError(msg)
        constructors_ids.append(constructor)

    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize(
            "constructor_eager", eager_constructors, ids=eager_constructors_ids
        )
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize("constructor", constructors, ids=constructors_ids)

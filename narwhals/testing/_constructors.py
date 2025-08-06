from __future__ import annotations

import os
import uuid
from copy import deepcopy
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, Callable, cast
from warnings import warn

import pytest

from narwhals._exceptions import find_stacklevel
from narwhals._utils import Implementation, generate_temporary_column_name, parse_version

if TYPE_CHECKING:
    import duckdb
    import ibis
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from ibis.backends.duckdb import Backend as IbisDuckDBBackend
    from pyspark.sql import DataFrame as PySparkDataFrame
    from typing_extensions import TypeAlias

    from narwhals._spark_like.dataframe import SQLFrameDataFrame
    from narwhals.typing import DataFrameLike, NativeFrame, NativeLazyFrame

    Data: TypeAlias = "dict[str, Any]"

    Constructor: TypeAlias = Callable[
        [Any], "NativeLazyFrame | NativeFrame | DataFrameLike"
    ]
    ConstructorEager: TypeAlias = Callable[[Any], "NativeFrame | DataFrameLike"]
    ConstructorLazy: TypeAlias = Callable[[Any], "NativeLazyFrame"]


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
    import pyarrow as pa

    duckdb.sql("""set timezone = 'UTC'""")

    _df = pa.table(obj)
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
    import pyarrow as pa

    table = pa.table(obj)
    table_name = str(uuid.uuid4())
    return _ibis_backend().create_table(table_name, table)


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
LAZY_CONSTRUCTORS: dict[str, ConstructorLazy] = {
    "dask": dask_lazy_p2_constructor,
    "polars[lazy]": polars_lazy_constructor,
    "duckdb": duckdb_lazy_constructor,
    "pyspark": pyspark_lazy_constructor,  # type: ignore[dict-item]
    "sqlframe": sqlframe_pyspark_lazy_constructor,
    "ibis": ibis_lazy_constructor,
}
GPU_CONSTRUCTORS: dict[str, ConstructorEager] = {"cudf": cudf_constructor}

MIN_PANDAS_NULLABLE_VERSION = (2,)


def get_module_version_as_tuple(module_name: str) -> tuple[int, ...]:
    try:
        return parse_version(__import__(module_name).__version__)
    except ImportError:
        return (0, 0, 0)


@cache
def backend_is_available(impl: Implementation) -> bool:
    try:
        impl._backend_version()
        backend_is_available = True

    except ValueError as exc:
        # ValueError is generated if the library is installed with a version lower than
        # the minimal supported by Narwhals
        warn(message=str(exc), category=UserWarning, stacklevel=find_stacklevel())
        backend_is_available = True

    except ModuleNotFoundError:
        backend_is_available = False

    return backend_is_available


PANDAS_VERSION = get_module_version_as_tuple("pandas")
PYARROW_AVAILABLE = backend_is_available(Implementation.PYARROW)


def get_constructors(
    selected_constructors: list[str],
) -> tuple[list[ConstructorEager], list[str], list[Constructor], list[str]]:
    eager_constructors: list[ConstructorEager] = []
    eager_ids: list[str] = []
    lazy_constructors: list[Constructor] = []
    lazy_ids: list[str] = []

    for _id in selected_constructors:
        if (
            _id in {"pandas[nullable]", "pandas[pyarrow]"}
            and MIN_PANDAS_NULLABLE_VERSION > PANDAS_VERSION
        ):
            continue  # pragma: no cover

        if _id in EAGER_CONSTRUCTORS:
            eager_constructors.append(EAGER_CONSTRUCTORS[_id])
            eager_ids.append(_id)
        elif _id in {"pyspark", "pyspark[connect]"}:  # pragma: no cover
            lazy_constructors.append(pyspark_lazy_constructor())
            lazy_ids.append(_id)
        elif _id in LAZY_CONSTRUCTORS:
            lazy_constructors.append(LAZY_CONSTRUCTORS[_id])
            lazy_ids.append(_id)
        else:  # pragma: no cover
            msg = f"Expected one of {EAGER_CONSTRUCTORS.keys()} or {LAZY_CONSTRUCTORS.keys()}, got {_id}"
            raise ValueError(msg)

    return eager_constructors, eager_ids, lazy_constructors, lazy_ids

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, cast

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.dependencies import get_cudf, get_modin
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals._spark_like.utils import SparkSession
    from narwhals._typing import LazyAllowed
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}


def test_lazy_to_default(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)

    expected_cls: Any
    if "polars" in str(constructor_eager):
        import polars as pl

        expected_cls = pl.LazyFrame
    elif "pandas" in str(constructor_eager):
        import pandas as pd

        expected_cls = pd.DataFrame
    elif "modin" in str(constructor_eager):
        mpd = get_modin()
        expected_cls = mpd.DataFrame
    elif "cudf" in str(constructor_eager):
        cudf = get_cudf()
        expected_cls = cudf.DataFrame
    else:  # pyarrow
        import pyarrow as pa

        expected_cls = pa.Table

    assert isinstance(result.to_native(), expected_cls)


@pytest.mark.parametrize(
    "backend",
    [
        Implementation.POLARS,
        Implementation.DUCKDB,
        Implementation.DASK,
        Implementation.IBIS,
        Implementation.PYSPARK,
        Implementation.SQLFRAME,
        "polars",
        "duckdb",
        "dask",
        "ibis",
        "pyspark",
        "sqlframe",
    ],
)
def test_lazy(constructor_eager: ConstructorEager, backend: LazyAllowed) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())

    if (
        is_spark_connect := os.environ.get("SPARK_CONNECT", None)
    ) is not None and impl.is_pyspark():
        impl = Implementation.PYSPARK_CONNECT

    df = nw.from_native(constructor_eager(data), eager_only=True)
    if (
        impl.is_duckdb()
        and df.implementation.is_pandas()
        and df.implementation._backend_version() >= (3, 0, 0)
    ):
        # Reason: https://github.com/duckdb/duckdb/issues/18297
        # > duckdb.duckdb.NotImplementedException: Not implemented Error: Data type 'str' not recognized
        return

    session: SparkSession | None
    if impl is Implementation.SQLFRAME:
        from sqlframe.duckdb import DuckDBSession

        session = DuckDBSession()
    elif impl in {Implementation.PYSPARK, Implementation.PYSPARK_CONNECT}:
        if is_spark_connect:
            from pyspark.sql.connect.session import SparkSession as PySparkSession
        else:
            from pyspark.sql import SparkSession as PySparkSession

        builder = cast("PySparkSession.Builder", PySparkSession.builder).appName(
            "unit-tests"
        )
        session = (  # pyright: ignore[reportAssignmentType]
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
    else:
        session = None

    result = df.lazy(backend=backend, session=session)
    assert isinstance(result, nw.LazyFrame)
    assert result.implementation == impl
    assert_equal_data(df.sort("a"), data)


@pytest.mark.parametrize("backend", ["pyspark", "sqlframe"])
def test_lazy_spark_like_requires_session(
    constructor_eager: ConstructorEager, backend: LazyAllowed
) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())

    df = nw.from_native(constructor_eager(data), eager_only=True)

    err_msg = re.escape("Spark like backends require `session` to be not None.")
    with pytest.raises(ValueError, match=err_msg):
        df.lazy(backend=backend, session=None)


def test_lazy_backend_invalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match="Not-supported backend"):
        df.lazy(backend=Implementation.PANDAS)  # type: ignore[arg-type]

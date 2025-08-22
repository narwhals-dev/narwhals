from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

pytest.importorskip("polars")
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._typing import EagerAllowed, _LazyOnly

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    fp = tmp_path_factory.mktemp("data") / "file.csv"
    filepath = str(fp)
    pl.DataFrame(data).write_csv(filepath)
    return filepath


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    fp = tmp_path_factory.mktemp("data") / "file.parquet"
    filepath = str(fp)
    pl.DataFrame(data).write_parquet(filepath)
    return filepath


def test_read_csv(csv_path: str, eager_backend: EagerAllowed) -> None:
    result = nw.read_csv(csv_path, backend=eager_backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw.DataFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_csv_kwargs(csv_path: str) -> None:
    result = nw.read_csv(csv_path, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@pytest.mark.parametrize("backend", ["duckdb", "ibis", "sqlframe"])
def test_read_csv_raise_with_lazy(csv_path: str, backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_csv(csv_path, backend=backend)  # type: ignore[arg-type]


def test_scan_csv(csv_path: str, constructor: Constructor) -> None:
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor):
        from sqlframe.duckdb import DuckDBSession

        kwargs = {"session": DuckDBSession(), "inferSchema": True, "header": True}
    elif "pyspark" in str(constructor):
        if is_spark_connect := os.environ.get("SPARK_CONNECT", None):
            from pyspark.sql.connect.session import SparkSession
        else:
            from pyspark.sql import SparkSession

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

        kwargs = {"session": session, "inferSchema": True, "header": True}

    else:
        kwargs = {}

    df = nw.from_native(constructor(data))
    backend = nw.get_native_namespace(df)
    result = nw.scan_csv(csv_path, backend=backend, **kwargs)
    assert_equal_data(result, data)
    assert isinstance(result, nw.LazyFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_csv_kwargs(csv_path: str) -> None:
    result = nw.scan_csv(csv_path, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_parquet(parquet_path: str, constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    result = nw.read_parquet(parquet_path, backend=backend)
    assert_equal_data(result, data)
    assert isinstance(result, nw.DataFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_read_parquet_kwargs(parquet_path: str) -> None:
    result = nw.read_parquet(parquet_path, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@pytest.mark.parametrize("backend", ["duckdb", "ibis", "sqlframe"])
def test_read_parquet_raise_with_lazy(parquet_path: str, backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_parquet(parquet_path, backend=backend)  # type: ignore[arg-type]


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_parquet(parquet_path: str, constructor: Constructor) -> None:
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor):
        from sqlframe.duckdb import DuckDBSession

        kwargs = {"session": DuckDBSession(), "inferSchema": True}

    elif "pyspark" in str(constructor):
        if is_spark_connect := os.environ.get("SPARK_CONNECT", None):
            from pyspark.sql.connect.session import SparkSession
        else:
            from pyspark.sql import SparkSession

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

        kwargs = {"session": session, "inferSchema": True, "header": True}
    else:
        kwargs = {}
    df = nw.from_native(constructor(data))
    backend = nw.get_native_namespace(df)
    result = nw.scan_parquet(parquet_path, backend=backend, **kwargs)
    assert_equal_data(result, data)
    assert isinstance(result, nw.LazyFrame)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_scan_parquet_kwargs(parquet_path: str) -> None:
    result = nw.scan_parquet(parquet_path, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@pytest.mark.parametrize("spark_like_backend", ["pyspark", "sqlframe"])
@pytest.mark.parametrize("scan_method", ["scan_csv", "scan_parquet"])
def test_scan_fail_spark_like_without_session(
    parquet_path: str,
    spark_like_backend: str,
    scan_method: Literal["scan_csv", "scan_parquet"],
) -> None:
    _ = pytest.importorskip(spark_like_backend)

    with pytest.raises(
        ValueError,
        match="Spark like backends require a session object to be passed in `kwargs`.",
    ):
        getattr(nw, scan_method)(parquet_path, backend=spark_like_backend)

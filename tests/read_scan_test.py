from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, assert_equal_data

pytest.importorskip("polars")
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from pyspark.sql import SparkSession
    from sqlframe.duckdb import DuckDBSession

    from narwhals._typing import EagerAllowed, _LazyOnly, _SparkLike

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}

skipif_pandas_lt_1_5 = pytest.mark.skipif(
    PANDAS_VERSION < (1, 5), reason="too old for pyarrow"
)
lazy_core_backend = pytest.mark.parametrize("backend", ["duckdb", "ibis", "sqlframe"])
spark_like_backend = pytest.mark.parametrize("backend", ["pyspark", "sqlframe"])


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


def assert_equal_eager(result: nw.DataFrame[Any]) -> None:
    assert_equal_data(result, data)
    assert isinstance(result, nw.DataFrame)


def assert_equal_lazy(result: nw.LazyFrame[Any]) -> None:
    assert_equal_data(result, data)
    assert isinstance(result, nw.LazyFrame)


def native_namespace(cb: Constructor, /) -> ModuleType:
    return nw.get_native_namespace(nw.from_native(cb(data)))  # type: ignore[no-any-return]


def test_read_csv(csv_path: str, eager_backend: EagerAllowed) -> None:
    result = nw.read_csv(csv_path, backend=eager_backend)
    assert_equal_eager(result)


@skipif_pandas_lt_1_5
def test_read_csv_kwargs(csv_path: str) -> None:
    result = nw.read_csv(csv_path, backend=pd, engine="pyarrow")
    assert_equal_eager(result)


@lazy_core_backend
def test_read_csv_raise_with_lazy(csv_path: str, backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_csv(csv_path, backend=backend)  # type: ignore[arg-type]


def sqlframe_session() -> DuckDBSession:
    from sqlframe.duckdb import DuckDBSession

    # NOTE: `__new__` override inferred by `pyright` only
    # https://github.com/eakmanrq/sqlframe/blob/772b3a6bfe5a1ffd569b7749d84bea2f3a314510/sqlframe/base/session.py#L181-L184
    return cast("DuckDBSession", DuckDBSession())  # type: ignore[redundant-cast]


def pyspark_session() -> SparkSession:
    if is_spark_connect := os.environ.get("SPARK_CONNECT", None):
        from pyspark.sql.connect.session import SparkSession
    else:
        from pyspark.sql import SparkSession
    builder = cast("SparkSession.Builder", SparkSession.builder).appName("unit-tests")
    builder = (
        builder.remote(f"sc://localhost:{os.environ.get('SPARK_PORT', '15002')}")
        if is_spark_connect
        else builder.master("local[1]").config("spark.ui.enabled", "false")
    )
    return (
        builder.config("spark.default.parallelism", "1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def test_scan_csv(csv_path: str, constructor: Constructor) -> None:
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor):
        kwargs = {"session": sqlframe_session(), "inferSchema": True, "header": True}
    elif "pyspark" in str(constructor):
        kwargs = {"session": pyspark_session(), "inferSchema": True, "header": True}
    else:
        kwargs = {}
    result = nw.scan_csv(csv_path, backend=native_namespace(constructor), **kwargs)
    assert_equal_lazy(result)


@skipif_pandas_lt_1_5
def test_scan_csv_kwargs(csv_path: str) -> None:
    result = nw.scan_csv(csv_path, backend=pd, engine="pyarrow")
    assert_equal_data(result, data)


@skipif_pandas_lt_1_5
def test_read_parquet(parquet_path: str, eager_backend: EagerAllowed) -> None:
    result = nw.read_parquet(parquet_path, backend=eager_backend)
    assert_equal_eager(result)


@skipif_pandas_lt_1_5
def test_read_parquet_kwargs(parquet_path: str) -> None:
    result = nw.read_parquet(parquet_path, backend=pd, engine="pyarrow")
    assert_equal_eager(result)


@lazy_core_backend
def test_read_parquet_raise_with_lazy(parquet_path: str, backend: _LazyOnly) -> None:
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="Expected eager backend, found"):
        nw.read_parquet(parquet_path, backend=backend)  # type: ignore[arg-type]


@skipif_pandas_lt_1_5
def test_scan_parquet(parquet_path: str, constructor: Constructor) -> None:
    kwargs: dict[str, Any]
    if "sqlframe" in str(constructor):
        kwargs = {"session": sqlframe_session(), "inferSchema": True}
    elif "pyspark" in str(constructor):
        kwargs = {"session": pyspark_session(), "inferSchema": True, "header": True}
    else:
        kwargs = {}
    backend = native_namespace(constructor)
    result = nw.scan_parquet(parquet_path, backend=backend, **kwargs)
    assert_equal_lazy(result)


@skipif_pandas_lt_1_5
def test_scan_parquet_kwargs(parquet_path: str) -> None:
    result = nw.scan_parquet(parquet_path, backend=pd, engine="pyarrow")
    assert_equal_lazy(result)


@spark_like_backend
@pytest.mark.parametrize("scan_method", ["scan_csv", "scan_parquet"])
def test_scan_fail_spark_like_without_session(
    parquet_path: str,
    backend: _SparkLike,
    scan_method: Literal["scan_csv", "scan_parquet"],
) -> None:
    pytest.importorskip(backend)
    with pytest.raises(
        ValueError,
        match="Spark like backends require a session object to be passed in `kwargs`.",
    ):
        getattr(nw, scan_method)(parquet_path, backend=backend)

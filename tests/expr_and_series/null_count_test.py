from __future__ import annotations

from typing import Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1.0, None, None, 3.0], "b": [1.0, None, 4.0, 5.0], "g": [1, 1, 1, 2]}


def assert_null_count_dtype(result: nw.LazyFrame[Any], constructor: Constructor) -> None:
    # Polars keeps its native `UInt32`.
    expected = nw.UInt32 if "polars" in str(constructor) else nw.Int64
    for schema in (result.collect_schema(), result.collect().schema):
        assert schema["a"] == expected
        assert schema["b"] == expected


def test_null_count_select(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a", "b").null_count())
    assert_equal_data(result, {"a": [2], "b": [1]})
    assert_null_count_dtype(result, constructor)


def test_null_count_with_columns(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3"):
        reason = "broadcasting aggregations requires DuckDB>=1.3"
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(constructor(data)).lazy()
    result = df.with_columns(nw.col("a", "b").null_count()).sort("g")
    expected = {"a": [2, 2, 2, 2], "b": [1, 1, 1, 1], "g": [1, 1, 1, 2]}
    assert_equal_data(result, expected)
    assert_null_count_dtype(result, constructor)


@pytest.mark.filterwarnings("ignore:Found complex group-by:UserWarning")
def test_null_count_group_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask")):
        reason = "`null_count` is not a supported group-by aggregation on these backends"
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(constructor(data)).lazy()
    result = df.group_by("g").agg(nw.col("a", "b").null_count()).sort("g")
    assert_equal_data(result, {"g": [1, 2], "a": [2, 0], "b": [1, 0]})
    assert_null_count_dtype(result, constructor)


def test_null_count_over(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor)
        for x in ("pandas", "modin", "cudf", "dask", "pyarrow_table")
    ):
        reason = "`null_count().over()` is not supported on these backends"
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3"):
        request.applymarker(pytest.mark.xfail(reason="`over` requires DuckDB>=1.3"))
    df = nw.from_native(constructor(data)).lazy()
    result = df.with_columns(nw.col("a", "b").null_count().over("g")).sort("g")
    expected = {"a": [2, 2, 2, 0], "b": [1, 1, 1, 0], "g": [1, 1, 1, 2]}
    assert_equal_data(result, expected)
    assert_null_count_dtype(result, constructor)


def test_null_count_empty(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data)).lazy().filter(nw.col("b") > 100)
    result = df.select(nw.col("a", "b").null_count())
    assert_equal_data(result, {"a": [0], "b": [0]})
    assert_null_count_dtype(result, constructor)


def test_null_count_series(constructor_eager: ConstructorEager) -> None:
    data = [1, 2, None]
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = series.null_count()
    assert result == 1

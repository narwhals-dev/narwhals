from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, assert_equal_data, is_windows

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_replace_time_zone(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
    ):
        pytest.skip()

    if any(x in str(constructor) for x in ("cudf", "pyspark", "ibis", "duckdb")):
        request.applymarker(pytest.mark.xfail)
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").dt.replace_time_zone("Asia/Kathmandu"))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone == "Asia/Kathmandu"
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T00:00+0545", "2020-01-02T00:00+0545"]}
    assert_equal_data(result_str, expected)


def test_replace_time_zone_none(constructor: Constructor) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
    ):
        pytest.skip()
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").dt.replace_time_zone(None))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone is None
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M"))
    expected = {"a": ["2020-01-01T00:00", "2020-01-02T00:00"]}
    assert_equal_data(result_str, expected)


def test_replace_time_zone_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor_eager) and is_windows())
        or ("pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2,))
        or ("modin_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2,))
    ):
        pytest.skip()
    if any(x in str(constructor_eager) for x in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].dt.replace_time_zone("Asia/Kathmandu"))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone == "Asia/Kathmandu"
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T00:00+0545", "2020-01-02T00:00+0545"]}
    assert_equal_data(result_str, expected)


def test_replace_time_zone_none_series(constructor_eager: ConstructorEager) -> None:
    if (
        ("pyarrow" in str(constructor_eager) and is_windows())
        or ("pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2,))
        or ("modin_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2,))
    ):
        pytest.skip()
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].dt.replace_time_zone(None))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone is None
    result_str = result.select(df["a"].dt.to_string("%Y-%m-%dT%H:%M"))
    expected = {"a": ["2020-01-01T00:00", "2020-01-02T00:00"]}
    assert_equal_data(result_str, expected)


def test_replace_time_zone_to_connection_tz_duckdb() -> None:
    pytest.importorskip("duckdb")

    import duckdb

    duckdb.sql("set timezone = 'Asia/Kolkata'")
    rel = duckdb.sql("""select * from values (timestamptz '2020-01-01') df(a)""")
    result = nw.from_native(rel).with_columns(
        nw.col("a").dt.replace_time_zone("Asia/Kolkata")
    )
    expected = {"a": [datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata"))]}
    assert_equal_data(result, expected)
    with pytest.raises(NotImplementedError):
        result = nw.from_native(rel).with_columns(
            nw.col("a").dt.replace_time_zone("Asia/Kathmandu")
        )


def test_replace_time_zone_to_connection_tz_pyspark(
    constructor: Constructor,
) -> None:  # pragma: no cover
    if "pyspark" not in str(constructor) or "sqlframe" in str(constructor):
        pytest.skip()
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.config(
        "spark.sql.session.timeZone", "UTC"
    ).getOrCreate()
    df = nw.from_native(
        session.createDataFrame([(datetime(2020, 1, 1, tzinfo=timezone.utc),)], ["a"])
    )
    result = nw.from_native(df).with_columns(nw.col("a").dt.replace_time_zone("UTC"))
    expected = {"a": [datetime(2020, 1, 1, tzinfo=timezone.utc)]}
    assert_equal_data(result, expected)
    with pytest.raises(NotImplementedError):
        result = nw.from_native(df).with_columns(
            nw.col("a").dt.replace_time_zone("Asia/Kathmandu")
        )

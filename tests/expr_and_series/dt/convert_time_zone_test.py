from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    assert_equal_data,
    is_windows,
)

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_convert_time_zone(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pyarrow_table" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
    ):
        pytest.skip()
    if any(x in str(constructor) for x in ("cudf", "duckdb", "pyspark", "ibis")):
        request.applymarker(pytest.mark.xfail)
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").dt.convert_time_zone("Asia/Kathmandu"))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone == "Asia/Kathmandu"
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T05:45+0545", "2020-01-02T05:45+0545"]}
    assert_equal_data(result_str, expected)


def test_convert_time_zone_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor_eager) and is_windows())
        or ("pyarrow_table" in str(constructor_eager) and is_windows())
        or ("pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1))
        or ("modin_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1))
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
    result = df.select(df["a"].dt.convert_time_zone("Asia/Kathmandu"))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone == "Asia/Kathmandu"
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T05:45+0545", "2020-01-02T05:45+0545"]}
    assert_equal_data(result_str, expected)


def test_convert_time_zone_from_none(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pyarrow_table" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
    ):
        pytest.skip()
    if any(x in str(constructor) for x in ("cudf", "duckdb", "pyspark", "ibis")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 7):
        # polars used to disallow this
        pytest.skip()
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").dt.replace_time_zone(None).dt.convert_time_zone("Asia/Kathmandu")
    )
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone == "Asia/Kathmandu"
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T05:45+0545", "2020-01-02T05:45+0545"]}
    assert_equal_data(result_str, expected)


def test_convert_time_zone_to_none(constructor: Constructor) -> None:
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor(data))
    with pytest.raises(TypeError, match="Target `time_zone` cannot be `None`"):
        df.select(nw.col("a").dt.convert_time_zone(None))  # type: ignore[arg-type]


def test_convert_time_zone_to_none_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor_eager(data))
    with pytest.raises(TypeError, match="Target `time_zone` cannot be `None`"):
        df["a"].dt.convert_time_zone(None)  # type: ignore[arg-type]


def test_convert_time_zone_to_connection_tz_duckdb() -> None:
    pytest.importorskip("duckdb")

    import duckdb

    duckdb.sql("set timezone = 'Asia/Kolkata'")
    rel = duckdb.sql("""select * from values (timestamptz '2020-01-01') df(a)""")
    result = nw.from_native(rel).with_columns(
        nw.col("a").dt.convert_time_zone("Asia/Kolkata")
    )
    expected = {"a": [datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata"))]}
    assert_equal_data(result, expected)
    with pytest.raises(NotImplementedError):
        result = nw.from_native(rel).with_columns(
            nw.col("a").dt.convert_time_zone("Asia/Kathmandu")
        )


def test_convert_time_zone_to_connection_tz_pyspark(
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
    result = nw.from_native(df).with_columns(nw.col("a").dt.convert_time_zone("UTC"))
    expected = {"a": [datetime(2020, 1, 1, tzinfo=timezone.utc)]}
    assert_equal_data(result, expected)
    with pytest.raises(NotImplementedError):
        result = nw.from_native(df).with_columns(
            nw.col("a").dt.convert_time_zone("Asia/Kathmandu")
        )

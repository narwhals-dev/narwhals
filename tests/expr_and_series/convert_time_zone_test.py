from datetime import datetime
from datetime import timezone
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import assert_equal_data
from tests.utils import is_windows


def test_convert_time_zone(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (any(x in str(constructor) for x in ("pyarrow", "modin")) and is_windows()) or (
        "pandas_pyarrow" in str(constructor) and parse_version(pd.__version__) < (2, 1)
    ):
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
    assert result_dtype.time_zone == "Asia/Kathmandu"  # type: ignore[attr-defined]
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T05:45+0545", "2020-01-02T05:45+0545"]}
    assert_equal_data(result_str, expected)


def test_convert_time_zone_series(
    constructor_eager: Any, request: pytest.FixtureRequest
) -> None:
    if (
        any(x in str(constructor_eager) for x in ("pyarrow", "modin")) and is_windows()
    ) or (
        "pandas_pyarrow" in str(constructor_eager)
        and parse_version(pd.__version__) < (2, 1)
    ):
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
    assert result_dtype.time_zone == "Asia/Kathmandu"  # type: ignore[attr-defined]
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T05:45+0545", "2020-01-02T05:45+0545"]}
    assert_equal_data(result_str, expected)


def test_convert_time_zone_from_none(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        (any(x in str(constructor) for x in ("pyarrow", "modin")) and is_windows())
        or (
            "pandas_pyarrow" in str(constructor)
            and parse_version(pd.__version__) < (2, 1)
        )
        or ("pyarrow_table" in str(constructor) and parse_version(pa.__version__) < (12,))
    ):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and parse_version(pl.__version__) < (0, 20, 7):
        # polars used to disallow this
        request.applymarker(pytest.mark.xfail)
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
    assert result_dtype.time_zone == "Asia/Kathmandu"  # type: ignore[attr-defined]
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


def test_convert_time_zone_to_none_series(constructor_eager: Any) -> None:
    data = {
        "a": [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
        ]
    }
    df = nw.from_native(constructor_eager(data))
    with pytest.raises(TypeError, match="Target `time_zone` cannot be `None`"):
        df["a"].dt.convert_time_zone(None)  # type: ignore[arg-type]

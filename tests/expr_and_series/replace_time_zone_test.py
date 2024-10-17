from datetime import datetime
from datetime import timezone
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import compare_dicts
from tests.utils import is_windows


def test_replace_time_zone(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        (any(x in str(constructor) for x in ("pyarrow", "modin")) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and parse_version(pd.__version__) < (2,))
        or ("pyarrow_table" in str(constructor) and parse_version(pa.__version__) < (12,))
        or ("cudf" in str(constructor))
    ):
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
    assert result_dtype.time_zone == "Asia/Kathmandu"  # type: ignore[attr-defined]
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T00:00+0545", "2020-01-02T00:00+0545"]}
    compare_dicts(result_str, expected)


def test_replace_time_zone_none(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        (any(x in str(constructor) for x in ("pyarrow", "modin")) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and parse_version(pd.__version__) < (2,))
        or ("pyarrow_table" in str(constructor) and parse_version(pa.__version__) < (12,))
    ):
        request.applymarker(pytest.mark.xfail)
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
    assert result_dtype.time_zone is None  # type: ignore[attr-defined]
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M"))
    expected = {"a": ["2020-01-01T00:00", "2020-01-02T00:00"]}
    compare_dicts(result_str, expected)


def test_replace_time_zone_series(
    constructor_eager: Any, request: pytest.FixtureRequest
) -> None:
    if (
        (any(x in str(constructor_eager) for x in ("pyarrow", "modin")) and is_windows())
        or (
            "pandas_pyarrow" in str(constructor_eager)
            and parse_version(pd.__version__) < (2,)
        )
        or (
            "pyarrow_table" in str(constructor_eager)
            and parse_version(pa.__version__) < (12,)
        )
        or ("cudf" in str(constructor_eager))
    ):
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
    assert result_dtype.time_zone == "Asia/Kathmandu"  # type: ignore[attr-defined]
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T00:00+0545", "2020-01-02T00:00+0545"]}
    compare_dicts(result_str, expected)


def test_replace_time_zone_none_series(
    constructor_eager: Any, request: pytest.FixtureRequest
) -> None:
    if (
        (any(x in str(constructor_eager) for x in ("pyarrow", "modin")) and is_windows())
        or (
            "pandas_pyarrow" in str(constructor_eager)
            and parse_version(pd.__version__) < (2,)
        )
        or (
            "pyarrow_table" in str(constructor_eager)
            and parse_version(pa.__version__) < (12,)
        )
    ):
        request.applymarker(pytest.mark.xfail)
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
    assert result_dtype.time_zone is None  # type: ignore[attr-defined]
    result_str = result.select(df["a"].dt.to_string("%Y-%m-%dT%H:%M"))
    expected = {"a": ["2020-01-01T00:00", "2020-01-02T00:00"]}
    compare_dicts(result_str, expected)

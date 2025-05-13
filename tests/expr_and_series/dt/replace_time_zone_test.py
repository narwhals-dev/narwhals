from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data
from tests.utils import is_windows

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_replace_time_zone(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,))
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
    result = df.select(nw.col("a").dt.replace_time_zone("Asia/Kathmandu"))
    result_dtype = result.collect_schema()["a"]
    assert result_dtype == nw.Datetime
    assert isinstance(result_dtype, nw.Datetime)
    assert result_dtype.time_zone == "Asia/Kathmandu"
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    expected = {"a": ["2020-01-01T00:00+0545", "2020-01-02T00:00+0545"]}
    assert_equal_data(result_str, expected)


def test_replace_time_zone_none(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,))
    ):
        pytest.skip()
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
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
        or ("pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (12,))
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
        or ("pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (12,))
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

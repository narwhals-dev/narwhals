from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data
from tests.utils import is_windows

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_convert_time_zone(
    constructor: Constructor,
    request: pytest.FixtureRequest,
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
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
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
        or ("pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,))
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

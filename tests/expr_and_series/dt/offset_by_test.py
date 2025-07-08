from __future__ import annotations

from datetime import datetime, timezone

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    is_windows,
)

data = {
    "a": [datetime(2021, 3, 1, 12, 34, 56, 49012), datetime(2020, 1, 2, 2, 4, 14, 715123)]
}

data_tz = {"a": [datetime(2024, 1, 1, tzinfo=timezone.utc)]}

data_dst = {"a": [datetime(2020, 10, 25, tzinfo=timezone.utc)]}


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        (
            "2us",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49014),
                datetime(2020, 1, 2, 2, 4, 14, 715125),
            ],
        ),
        (
            "2ms",
            [
                datetime(2021, 3, 1, 12, 34, 56, 51012),
                datetime(2020, 1, 2, 2, 4, 14, 717123),
            ],
        ),
        (
            "10s",
            [
                datetime(2021, 3, 1, 12, 35, 6, 49012),
                datetime(2020, 1, 2, 2, 4, 24, 715123),
            ],
        ),
        (
            "7m",
            [
                datetime(2021, 3, 1, 12, 41, 56, 49012),
                datetime(2020, 1, 2, 2, 11, 14, 715123),
            ],
        ),
        (
            "7h",
            [
                datetime(2021, 3, 1, 19, 34, 56, 49012),
                datetime(2020, 1, 2, 9, 4, 14, 715123),
            ],
        ),
        (
            "13d",
            [
                datetime(2021, 3, 14, 12, 34, 56, 49012),
                datetime(2020, 1, 15, 2, 4, 14, 715123),
            ],
        ),
        (
            "3mo",
            [
                datetime(2021, 6, 1, 12, 34, 56, 49012),
                datetime(2020, 4, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "2q",
            [
                datetime(2021, 9, 1, 12, 34, 56, 49012),
                datetime(2020, 7, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "3y",
            [
                datetime(2024, 3, 1, 12, 34, 56, 49012),
                datetime(2023, 1, 2, 2, 4, 14, 715123),
            ],
        ),
    ],
)
def test_offset_by(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    by: str,
    expected: list[datetime],
) -> None:
    df = nw.from_native(constructor(data))
    if df.implementation.is_pyspark_connect():
        # missing feature upstream
        request.applymarker(pytest.mark.xfail())
    if by == "2q" and "sqlframe" in str(constructor):
        # https://github.com/eakmanrq/sqlframe/issues/443
        request.applymarker(pytest.mark.xfail())
    if any(x in by for x in ("y", "q", "mo")) and any(
        x in str(constructor) for x in ("dask", "pyarrow", "ibis")
    ):
        request.applymarker(pytest.mark.xfail())
    if by.endswith("d") and any(x in str(constructor) for x in ("dask", "ibis")):
        request.applymarker(pytest.mark.xfail())
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/19292
        request.applymarker(pytest.mark.xfail)
    result = df.select(nw.col("a").dt.offset_by(by))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("2d", ["2024-01-03T05:45+0545"]),
        ("5mo", ["2024-06-01T05:45+0545"]),
        ("7q", ["2025-10-01T05:45+0545"]),
        ("5y", ["2029-01-01T05:45+0545"]),
    ],
)
def test_offset_by_tz(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    by: str,
    expected: list[datetime],
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pyarrow_table" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
        or ("pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,))
    ):
        pytest.skip()
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "sqlframe", "ibis")):
        # pyspark,duckdb don't support changing time zones.
        # convert_time_zone is not supported for ibis.
        request.applymarker(pytest.mark.xfail())
    if any(x in by for x in ("y", "q", "mo")) and any(
        x in str(constructor) for x in ("dask", "pyarrow", "ibis")
    ):
        request.applymarker(pytest.mark.xfail())
    if by.endswith("d") and any(x in str(constructor) for x in ("dask",)):
        request.applymarker(pytest.mark.xfail())
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/19292
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data_tz))
    df = df.select(nw.col("a").dt.convert_time_zone("Asia/Kathmandu"))
    result = df.select(nw.col("a").dt.offset_by(by))
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    assert_equal_data(result_str, {"a": expected})


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("2d", ["2020-10-27T02:00+0100"]),
        ("5mo", ["2021-03-25T02:00+0100"]),
        ("1q", ["2021-01-25T02:00+0100"]),
    ],
)
def test_offset_by_dst(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    by: str,
    expected: list[datetime],
) -> None:
    if (
        ("pyarrow" in str(constructor) and is_windows())
        or ("pyarrow_table" in str(constructor) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
        or ("modin_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1))
        or ("pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,))
    ):
        pytest.skip()
    if any(
        x in str(constructor) for x in ("duckdb", "sqlframe", "pyspark", "ibis", "cudf")
    ):
        # pyspark,duckdb don't support changing time zones.
        # convert_time_zone is not supported for ibis.
        # cudf https://github.com/rapidsai/cudf/issues/19292
        request.applymarker(pytest.mark.xfail())
    if any(x in by for x in ("y", "q", "mo")) and any(
        x in str(constructor) for x in ("dask", "pyarrow")
    ):
        request.applymarker(pytest.mark.xfail())
    if by.endswith("d") and any(x in str(constructor) for x in ("dask",)):
        request.applymarker(pytest.mark.xfail())
    df = nw.from_native(constructor(data_dst))
    df = df.with_columns(a=nw.col("a").dt.convert_time_zone("Europe/Amsterdam"))
    result = df.select(nw.col("a").dt.offset_by(by))
    result_str = result.select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M%z"))
    assert_equal_data(result_str, {"a": expected})


def test_offset_by_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # https://github.com/rapidsai/cudf/issues/19292
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].dt.offset_by("1h"))
    expected = {
        "a": [
            datetime(2021, 3, 1, 13, 34, 56, 49012),
            datetime(2020, 1, 2, 3, 4, 14, 715123),
        ]
    }
    assert_equal_data(result, expected)


def test_offset_by_invalid_interval(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/19292
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    msg = "Invalid `every` string"
    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").dt.offset_by("1r"))

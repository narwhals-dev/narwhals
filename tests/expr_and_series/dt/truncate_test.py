from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {
    "a": [datetime(2021, 3, 1, 12, 34, 56, 49012), datetime(2020, 1, 2, 2, 4, 14, 715123)]
}


@pytest.mark.parametrize(
    ("every", "expected"),
    [
        (
            "1ns",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49012),
                datetime(2020, 1, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "1us",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49012),
                datetime(2020, 1, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "1ms",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49000),
                datetime(2020, 1, 2, 2, 4, 14, 715000),
            ],
        ),
        ("1s", [datetime(2021, 3, 1, 12, 34, 56), datetime(2020, 1, 2, 2, 4, 14)]),
        ("1m", [datetime(2021, 3, 1, 12, 34), datetime(2020, 1, 2, 2, 4)]),
        ("1h", [datetime(2021, 3, 1, 12, 0, 0, 0), datetime(2020, 1, 2, 2, 0, 0, 0)]),
        ("1d", [datetime(2021, 3, 1), datetime(2020, 1, 2)]),
        ("1mo", [datetime(2021, 3, 1), datetime(2020, 1, 1)]),
        ("1q", [datetime(2021, 1, 1), datetime(2020, 1, 1)]),
        ("1y", [datetime(2021, 1, 1), datetime(2020, 1, 1)]),
    ],
)
def test_truncate(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    every: str,
    expected: list[datetime],
) -> None:
    if every.endswith("ns") and any(
        x in str(constructor) for x in ("polars", "duckdb", "pyspark", "ibis")
    ):
        request.applymarker(pytest.mark.xfail())
    if any(every.endswith(x) for x in ("mo", "q", "y")) and any(
        x in str(constructor) for x in ("dask", "cudf")
    ):
        request.applymarker(pytest.mark.xfail(reason="Not implemented"))
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").dt.truncate(every))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("every", "expected"),
    [
        (
            "2ns",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49012),
                datetime(2020, 1, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "2us",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49012),
                datetime(2020, 1, 2, 2, 4, 14, 715122),
            ],
        ),
        (
            "2ms",
            [
                datetime(2021, 3, 1, 12, 34, 56, 48000),
                datetime(2020, 1, 2, 2, 4, 14, 714000),
            ],
        ),
        ("10s", [datetime(2021, 3, 1, 12, 34, 50), datetime(2020, 1, 2, 2, 4, 10)]),
        ("7m", [datetime(2021, 3, 1, 12, 30), datetime(2020, 1, 2, 1, 59)]),
        ("7h", [datetime(2021, 3, 1, 9, 0, 0, 0), datetime(2020, 1, 2, 0, 0, 0, 0)]),
        ("13d", [datetime(2021, 2, 23), datetime(2019, 12, 22)]),
        ("3mo", [datetime(2021, 1, 1), datetime(2020, 1, 1)]),
        ("2q", [datetime(2021, 1, 1), datetime(2020, 1, 1)]),
    ],
)
def test_truncate_multiples(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    every: str,
    expected: list[datetime],
) -> None:
    if any(x in str(constructor) for x in ("cudf", "pyspark", "duckdb")):
        # Reasons:
        # - cudf: https://github.com/rapidsai/cudf/issues/18654
        # - pyspark/sqlframe: Only multiple 1 is currently supported
        request.applymarker(pytest.mark.xfail())
    if every.endswith("ns") and any(
        x in str(constructor) for x in ("polars", "duckdb", "ibis")
    ):
        request.applymarker(pytest.mark.xfail())
    if any(every.endswith(x) for x in ("mo", "q", "y")) and any(
        x in str(constructor) for x in ("dask",)
    ):
        request.applymarker(pytest.mark.xfail(reason="Not implemented"))
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").dt.truncate(every))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("every", "expected"),
    [
        (
            "1ns",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49012),
                datetime(2020, 1, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "7ns",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49012),
                datetime(2020, 1, 2, 2, 4, 14, 715122),
            ],
        ),
    ],
)
def test_truncate_polars_ns(every: str, expected: list[datetime]) -> None:
    pytest.importorskip("polars")

    import polars as pl

    df_pl = pl.DataFrame(data, schema={"a": pl.Datetime(time_unit="ns")})
    df = nw.from_native(df_pl)
    result = df.select(nw.col("a").dt.truncate(every))
    assert_equal_data(result, {"a": expected})


def test_truncate_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].dt.truncate("1h"))
    expected = {
        "a": [datetime(2021, 3, 1, 12, 0, 0, 0), datetime(2020, 1, 2, 2, 0, 0, 0)]
    }
    assert_equal_data(result, expected)


def test_truncate_invalid_interval(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    msg = "Invalid `every` string"
    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").dt.truncate("1r"))


def test_truncate_invalid_multiple(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    msg = "Only the following multiples are supported"
    msg_year = "Only multiple 1 is currently supported for 'y' unit"
    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").dt.truncate("5mo"))
    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").dt.truncate("5q"))
    with pytest.raises(ValueError, match=msg_year):
        df.select(nw.col("a").dt.truncate("5y"))


def test_pandas_numpy_nat() -> None:
    # The pandas implementation goes via NumPy, so check NaT are preserved.
    df = nw.from_native(
        pd.DataFrame({"a": [datetime(2020, 1, 1), None, datetime(2020, 1, 2)]})
    )
    result: nw.DataFrame[pd.DataFrame] = df.select(nw.col("a").dt.truncate("1mo"))
    expected = {"a": [datetime(2020, 1, 1), None, datetime(2020, 1, 1)]}
    assert_equal_data(result, expected)
    assert result.item(1, 0) is pd.NaT


def test_truncate_tz_aware_duckdb() -> None:
    pytest.importorskip("duckdb")

    import duckdb

    duckdb.sql("""set timezone = 'Europe/Amsterdam'""")
    rel = duckdb.sql("""select * from values (timestamptz '2020-10-25') df(a)""")
    result = nw.from_native(rel).with_columns(a_truncated=nw.col("a").dt.truncate("1mo"))
    expected = {
        "a": [datetime(2020, 10, 25, tzinfo=ZoneInfo("Europe/Amsterdam"))],
        "a_truncated": [datetime(2020, 10, 1, tzinfo=ZoneInfo("Europe/Amsterdam"))],
    }
    assert_equal_data(result, expected)
    duckdb.sql("""set timezone = 'UTC'""")

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

import narwhals as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56, 49012),
        datetime(2020, 1, 2, 2, 4, 14, 715123),
    ],
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
        (
            "1s",
            [
                datetime(2021, 3, 1, 12, 34, 56),
                datetime(2020, 1, 2, 2, 4, 14),
            ],
        ),
        (
            "1m",
            [
                datetime(2021, 3, 1, 12, 34),
                datetime(2020, 1, 2, 2, 4),
            ],
        ),
        (
            "1h",
            [
                datetime(2021, 3, 1, 12, 0, 0, 0),
                datetime(2020, 1, 2, 2, 0, 0, 0),
            ],
        ),
        (
            "1d",
            [
                datetime(2021, 3, 1),
                datetime(2020, 1, 2),
            ],
        ),
    ],
)
def test_truncate(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    every: str,
    expected: list[datetime],
) -> None:
    if any(x in str(constructor) for x in ("pyspark",)):
        request.applymarker(pytest.mark.xfail(reason="Bug"))
    if every.endswith("ns") and any(x in str(constructor) for x in ("polars", "duckdb")):
        request.applymarker(pytest.mark.xfail())
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
        (
            "10s",
            [
                datetime(2021, 3, 1, 12, 34, 50),
                datetime(2020, 1, 2, 2, 4, 10),
            ],
        ),
        (
            "7m",
            [
                datetime(2021, 3, 1, 12, 30),
                datetime(2020, 1, 2, 1, 59),
            ],
        ),
        (
            "1h",
            [
                datetime(2021, 3, 1, 12, 0, 0, 0),
                datetime(2020, 1, 2, 2, 0, 0, 0),
            ],
        ),
        (
            "13d",
            [
                datetime(2021, 2, 23),
                datetime(2019, 12, 22),
            ],
        ),
    ],
)
def test_truncate_custom(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    every: str,
    expected: list[datetime],
) -> None:
    if any(x in str(constructor) for x in ("pyspark",)):
        request.applymarker(pytest.mark.xfail(reason="Bug"))
    if every.endswith("ns") and any(x in str(constructor) for x in ("polars", "duckdb")):
        request.applymarker(pytest.mark.xfail())
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
    df_pl = pl.DataFrame(data, schema={"a": pl.Datetime(time_unit="ns")})
    df = nw.from_native(df_pl)
    result = df.select(nw.col("a").dt.truncate(every))
    assert_equal_data(result, {"a": expected})

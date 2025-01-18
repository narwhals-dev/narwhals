from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
import narwhals.stable.v1.selectors as ncs
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
}


def test_selectors(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(ncs.by_dtype([nw.Int64, nw.Float64]) + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_numeric(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(ncs.numeric() + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_boolean(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(ncs.boolean())
    expected = {"d": [True, False, True]}
    assert_equal_data(result, expected)


def test_string(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(ncs.string())
    expected = {"b": ["a", "b", "c"]}
    assert_equal_data(result, expected)


def test_categorical(
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if "pyarrow_table_constructor" in str(constructor) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    expected = {"b": ["a", "b", "c"]}

    df = nw.from_native(constructor(data)).with_columns(nw.col("b").cast(nw.Categorical))
    result = df.select(ncs.categorical())
    assert_equal_data(result, expected)


def test_datetime(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if (
        "pyspark" in str(constructor)
        or "duckdb" in str(constructor)
        or "dask" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,):
        request.applymarker(pytest.mark.xfail)

    ts1 = datetime(2000, 11, 20, 18, 12, 16, 600000)
    ts2 = datetime(2020, 10, 30, 10, 20, 25, 123000)

    data = {
        "numeric": [3.14, 6.28],
        "ts": [ts1, ts2],
    }
    time_units: list[Literal["ns", "us", "ms", "s"]] = ["ms", "us", "ns"]

    df = nw.from_native(constructor(data)).select(
        nw.col("numeric"),
        *[
            nw.col("ts").cast(nw.Datetime(time_unit=tu)).alias(f"ts_{tu}")
            for tu in time_units
        ],
        *[
            nw.col("ts")
            .dt.convert_time_zone("Europe/Lisbon")
            .cast(nw.Datetime(time_zone="Europe/Lisbon", time_unit=tu))
            .alias(f"ts_lisbon_{tu}")
            for tu in time_units
        ],
        *[
            nw.col("ts")
            .dt.convert_time_zone("Europe/Berlin")
            .cast(nw.Datetime(time_zone="Europe/Berlin", time_unit=tu))
            .alias(f"ts_berlin_{tu}")
            for tu in time_units
        ],
    )

    assert df.select(ncs.datetime()).collect_schema().names() == [
        "ts_ms",
        "ts_us",
        "ts_ns",
        "ts_lisbon_ms",
        "ts_lisbon_us",
        "ts_lisbon_ns",
        "ts_berlin_ms",
        "ts_berlin_us",
        "ts_berlin_ns",
    ]
    assert df.select(ncs.datetime(time_unit="ms")).collect_schema().names() == [
        "ts_ms",
        "ts_lisbon_ms",
        "ts_berlin_ms",
    ]
    assert df.select(ncs.datetime(time_unit=["us", "ns"])).collect_schema().names() == [
        "ts_us",
        "ts_ns",
        "ts_lisbon_us",
        "ts_lisbon_ns",
        "ts_berlin_us",
        "ts_berlin_ns",
    ]

    assert df.select(ncs.datetime(time_zone=None)).collect_schema().names() == [
        "ts_ms",
        "ts_us",
        "ts_ns",
    ]
    assert df.select(ncs.datetime(time_zone="*")).collect_schema().names() == [
        "ts_lisbon_ms",
        "ts_lisbon_us",
        "ts_lisbon_ns",
        "ts_berlin_ms",
        "ts_berlin_us",
        "ts_berlin_ns",
    ]
    assert df.select(
        ncs.datetime(time_zone=[None, "Europe/Berlin"])
    ).collect_schema().names() == [
        "ts_ms",
        "ts_us",
        "ts_ns",
        "ts_berlin_ms",
        "ts_berlin_us",
        "ts_berlin_ns",
    ]

    assert df.select(
        ncs.datetime(time_unit="ns", time_zone=[None, "Europe/Berlin"])
    ).collect_schema().names() == ["ts_ns", "ts_berlin_ns"]
    assert df.select(
        ncs.datetime(time_unit=["ms", "us"], time_zone=[None, "Europe/Berlin"])
    ).collect_schema().names() == ["ts_ms", "ts_us", "ts_berlin_ms", "ts_berlin_us"]


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (ncs.numeric() | ncs.boolean(), ["a", "c", "d"]),
        (ncs.numeric() & ncs.boolean(), []),
        (ncs.numeric() & ncs.by_dtype(nw.Int64), ["a"]),
        (ncs.numeric() | ncs.by_dtype(nw.Int64), ["a", "c"]),
        (~ncs.numeric(), ["b", "d"]),
        (ncs.boolean() & True, ["d"]),
        (ncs.boolean() | True, ["d"]),
        (ncs.numeric() - 1, ["a", "c"]),
        (ncs.all(), ["a", "b", "c", "d"]),
    ],
)
def test_set_ops(
    constructor: Constructor,
    selector: nw.selectors.Selector,
    expected: list[str],
    request: pytest.FixtureRequest,
) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(selector).collect_schema().names()
    assert sorted(result) == expected


@pytest.mark.parametrize("invalid_constructor", [pd.DataFrame, pa.table])
def test_set_ops_invalid(
    invalid_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(invalid_constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(invalid_constructor(data))
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 - ncs.numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 | ncs.numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 & ncs.numeric())

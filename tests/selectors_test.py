from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

import pytest

import narwhals as nw
import narwhals.selectors as ncs
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    assert_equal_data,
    is_windows,
)

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
}

data_regex = {"foo": ["x", "y"], "bar": [123, 456], "baz": [2.0, 5.5], "zap": [0, 1]}


def test_selectors(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(ncs.by_dtype([nw.Int64, nw.Float64]) + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_matches(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data_regex))
    result = df.select(ncs.matches("[^z]a") + 1)
    expected = {"bar": [124, 457], "baz": [3.0, 6.5]}
    assert_equal_data(result, expected)


def test_numeric(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(ncs.numeric() + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_boolean(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(ncs.boolean())
    expected = {"d": [True, False, True]}
    assert_equal_data(result, expected)


def test_string(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(ncs.string())
    expected = {"b": ["a", "b", "c"]}
    assert_equal_data(result, expected)


def test_categorical(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table_constructor" in str(constructor) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if (
        "pyspark" in str(constructor)
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
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
        or ("pyarrow" in str(constructor) and is_windows())
        or ("pandas" in str(constructor) and PANDAS_VERSION < (2,))
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        pytest.skip(reason="too slow")

    ts1 = datetime(2000, 11, 20, 18, 12, 16, 600000)
    ts2 = datetime(2020, 10, 30, 10, 20, 25, 123000)

    data = {"numeric": [3.14, 6.28], "ts": [ts1, ts2]}
    time_units: list[Literal["ns", "us", "ms", "s"]] = ["ms", "us", "ns"]

    df = nw.from_native(constructor(data)).select(
        nw.col("numeric"),
        *[
            nw.col("ts").cast(nw.Datetime(time_unit=tu)).alias(f"ts_{tu}")
            for tu in time_units
        ],
        *[
            nw.col("ts")
            .dt.replace_time_zone("Europe/Lisbon")
            .cast(nw.Datetime(time_zone="Europe/Lisbon", time_unit=tu))
            .alias(f"ts_lisbon_{tu}")
            for tu in time_units
        ],
        *[
            nw.col("ts")
            .dt.replace_time_zone("Europe/Berlin")
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

    assert df.with_columns(
        nw.col("ts_ms").dt.replace_time_zone("UTC").alias("ts_utc")
    ).select(
        ncs.datetime(time_unit=["ms", "us"], time_zone=[None, timezone.utc])
    ).collect_schema().names() == ["ts_ms", "ts_us", "ts_utc"]


def test_datetime_no_tz(constructor: Constructor) -> None:
    ts1 = datetime(2000, 11, 20, 18, 12, 16, 600000)
    ts2 = datetime(2020, 10, 30, 10, 20, 25, 123000)

    data = {"numeric": [3.14, 6.28], "ts": [ts1, ts2]}

    df = nw.from_native(constructor(data))
    assert df.select(ncs.datetime()).collect_schema().names() == ["ts"]


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
    if (
        any(x in str(constructor) for x in ("duckdb", "sqlframe", "ibis"))
        and not expected
    ):
        # https://github.com/narwhals-dev/narwhals/issues/2469
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(selector).collect_schema().names()
    assert sorted(result) == expected


def test_subtract_expr(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 27):
        # In old Polars versions, cs.numeric() - col('a')
        # would exclude column 'a' from the result, as opposed to
        # subtracting it.
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.select(ncs.numeric() - nw.col("a"))
    expected = {"a": [0, 0, 0], "c": [3.1, 4.0, 4.0]}
    assert_equal_data(result, expected)


def test_set_ops_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 - ncs.numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 | ncs.numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 & ncs.numeric())

    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for op: ('Selector' + 'Selector')"),
    ):
        df.select(ncs.boolean() + ncs.numeric())


@pytest.mark.skipif(is_windows(), reason="windows is what it is")
def test_tz_aware(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 19):
        # bug in old polars
        pytest.skip()
    if (
        "duckdb" in str(constructor)
        or "pyspark" in str(constructor)
        or "ibis" in str(constructor)
    ):
        # replace_time_zone not implemented
        request.applymarker(pytest.mark.xfail)

    data = {"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "c": [4, 5]}
    df = nw.from_native(constructor(data)).with_columns(
        b=nw.col("a").dt.replace_time_zone("Asia/Katmandu")
    )
    result = df.select(nw.selectors.by_dtype(nw.Datetime)).collect_schema().names()
    expected = ["a", "b"]
    assert result == expected
    result = df.select(nw.selectors.by_dtype(nw.Int64())).collect_schema().names()
    expected = ["c"]
    assert result == expected

from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import Literal

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56, 49000),
        datetime(2020, 1, 2, 2, 4, 14, 715000),
    ],
}


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("date", [date(2021, 3, 1), date(2020, 1, 2)]),
        ("year", [2021, 2020]),
        ("month", [3, 1]),
        ("day", [1, 2]),
        ("hour", [12, 2]),
        ("minute", [34, 4]),
        ("second", [56, 14]),
        ("millisecond", [49, 715]),
        ("microsecond", [49000, 715000]),
        ("nanosecond", [49000000, 715000000]),
        ("ordinal_day", [60, 2]),
    ],
)
def test_datetime_attributes(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    attribute: str,
    expected: list[int],
) -> None:
    if (
        attribute == "date"
        and "pandas" in str(constructor)
        and "pyarrow" not in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    if attribute == "date" and "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a").dt, attribute)())
    compare_dicts(result, {"a": expected})


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("date", [date(2021, 3, 1), date(2020, 1, 2)]),
        ("year", [2021, 2020]),
        ("month", [3, 1]),
        ("day", [1, 2]),
        ("hour", [12, 2]),
        ("minute", [34, 4]),
        ("second", [56, 14]),
        ("millisecond", [49, 715]),
        ("microsecond", [49000, 715000]),
        ("nanosecond", [49000000, 715000000]),
        ("ordinal_day", [60, 2]),
    ],
)
def test_datetime_attributes_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    attribute: str,
    expected: list[int],
) -> None:
    if (
        attribute == "date"
        and "pandas" in str(constructor_eager)
        and "pyarrow" not in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)
    if attribute == "date" and "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(getattr(df["a"].dt, attribute)())
    compare_dicts(result, {"a": expected})


def test_datetime_chained_attributes(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pandas" in str(constructor_eager) and "pyarrow" not in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].dt.date().dt.year())
    compare_dicts(result, {"a": [2021, 2020]})

    result = df.select(nw.col("a").dt.date().dt.year())
    compare_dicts(result, {"a": [2021, 2020]})


@pytest.mark.parametrize(
    ("original_time_unit", "time_unit", "expected"),
    [
        ("ns", "ns", [978307200000000000, None, 978480000000000000]),
        ("ns", "us", [978307200000000, None, 978480000000000]),
        ("ns", "ms", [978307200000, None, 978480000000]),
        ("us", "ns", [978307200000000000, None, 978480000000000000]),
        ("us", "us", [978307200000000, None, 978480000000000]),
        ("us", "ms", [978307200000, None, 978480000000]),
        ("ms", "ns", [978307200000000000, None, 978480000000000000]),
        ("ms", "us", [978307200000000, None, 978480000000000]),
        ("ms", "ms", [978307200000, None, 978480000000]),
        ("s", "ns", [978307200000000000, None, 978480000000000000]),
        ("s", "us", [978307200000000, None, 978480000000000]),
        ("s", "ms", [978307200000, None, 978480000000]),
    ],
)
def test_timestamp_datetimes(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    original_time_unit: Literal["us", "ns", "ms", "s"],
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if original_time_unit == "s" and "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    datetimes = {"a": [datetime(2001, 1, 1), None, datetime(2001, 1, 3)]}
    df = nw.from_native(constructor(datetimes))
    result = df.select(
        nw.col("a").cast(nw.Datetime(original_time_unit)).dt.timestamp(time_unit)
    )
    compare_dicts(result, {"a": expected})


@pytest.mark.parametrize(
    ("time_unit", "expected"),
    [
        ("ns", [978307200000000000, None, 978480000000000000]),
        ("us", [978307200000000, None, 978480000000000]),
        ("ms", [978307200000, None, 978480000000]),
    ],
)
def test_timestamp_dates(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(
        x in str(constructor)
        for x in ("pandas_constructor", "pandas_nullable_constructor", "cudf")
    ):
        request.applymarker(pytest.mark.xfail)

    dates = {"a": [datetime(2001, 1, 1), None, datetime(2001, 1, 3)]}
    if "dask" in str(constructor):
        df = nw.from_native(
            constructor(dates).astype({"a": "timestamp[ns][pyarrow]"})  # type: ignore[union-attr]
        )
    else:
        df = nw.from_native(constructor(dates))
    result = df.select(nw.col("a").dt.date().dt.timestamp(time_unit))
    compare_dicts(result, {"a": expected})


def test_timestamp_invalid_date(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data_str = {"a": ["x", "y", None]}
    data_num = {"a": [1, 2, None]}
    df_str = nw.from_native(constructor(data_str))
    df_num = nw.from_native(constructor(data_num))
    msg = "Input should be either of Date or Datetime type"
    with pytest.raises(TypeError, match=msg):
        df_str.select(nw.col("a").dt.timestamp())
    with pytest.raises(TypeError, match=msg):
        df_num.select(nw.col("a").dt.timestamp())


def test_timestamp_invalid_unit_expr(constructor: Constructor) -> None:
    time_unit_invalid = "i"
    msg = (
        "invalid `time_unit`"
        f"\n\nExpected one of {{'ns', 'us', 'ms'}}, got {time_unit_invalid!r}."
    )
    with pytest.raises(ValueError, match=msg):
        nw.from_native(constructor(data)).select(
            nw.col("a").dt.timestamp(time_unit_invalid)  # type: ignore[arg-type]
        )


def test_timestamp_invalid_unit_series(constructor_eager: ConstructorEager) -> None:
    time_unit_invalid = "i"
    msg = (
        "invalid `time_unit`"
        f"\n\nExpected one of {{'ns', 'us', 'ms'}}, got {time_unit_invalid!r}."
    )
    with pytest.raises(ValueError, match=msg):
        nw.from_native(constructor_eager(data))["a"].dt.timestamp(time_unit_invalid)  # type: ignore[arg-type]


def test_to_date(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(
        x in str(constructor)
        for x in ("pandas_constructor", "pandas_nullable_constructor")
    ):
        request.applymarker(pytest.mark.xfail)
    dates = {"a": [datetime(2001, 1, 1), None, datetime(2001, 1, 3)]}
    if "dask" in str(constructor):
        df = nw.from_native(constructor(dates).astype({"a": "timestamp[ns][pyarrow]"}))  # type: ignore[union-attr]
    else:
        df = nw.from_native(constructor(dates))
    result = df.select(nw.col("a").dt.date())
    assert result.collect_schema() == {"a": nw.Date}

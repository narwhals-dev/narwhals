from __future__ import annotations

from datetime import datetime
from typing import Literal

import hypothesis.strategies as st
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data
from tests.utils import is_windows

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56, 49000),
        datetime(2020, 1, 2, 2, 4, 14, 715000),
    ],
}


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
    constructor: ConstructorEager,
    original_time_unit: Literal["us", "ns", "ms", "s"],
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if original_time_unit == "s" and "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (
        2,
        2,
    ):  # pragma: no cover
        # pyarrow-backed timestamps were too inconsistent and unreliable before 2.2
        request.applymarker(pytest.mark.xfail(strict=False))
    datetimes = {"a": [datetime(2001, 1, 1), None, datetime(2001, 1, 3)]}
    df = nw.from_native(constructor(datetimes))
    result = df.select(
        nw.col("a").cast(nw.Datetime(original_time_unit)).dt.timestamp(time_unit)
    )
    assert_equal_data(result, {"a": expected})


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
def test_timestamp_datetimes_tz_aware(
    request: pytest.FixtureRequest,
    constructor: ConstructorEager,
    original_time_unit: Literal["us", "ns", "ms", "s"],
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if (
        (any(x in str(constructor) for x in ("pyarrow",)) and is_windows())
        or ("pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2,))
        or ("pyarrow_table" in str(constructor) and PYARROW_VERSION < (12,))
    ):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (
        2,
        2,
    ):  # pragma: no cover
        # pyarrow-backed timestamps were too inconsistent and unreliable before 2.2
        request.applymarker(pytest.mark.xfail(strict=False))
    if "dask" in str(constructor) and PANDAS_VERSION < (
        2,
        1,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    if original_time_unit == "s" and "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    datetimes = {"a": [datetime(2001, 1, 1), None, datetime(2001, 1, 3)]}
    df = nw.from_native(constructor(datetimes))
    result = df.select(
        nw.col("a")
        .cast(nw.Datetime(original_time_unit))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Asia/Kathmandu")
        .dt.timestamp(time_unit)
    )
    assert_equal_data(result, {"a": expected})


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
    constructor: ConstructorEager,
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
    if any(
        x in str(constructor)
        for x in (
            "pandas_constructor",
            "pandas_nullable_constructor",
            "cudf",
            "modin_constructor",
        )
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
    assert_equal_data(result, {"a": expected})


def test_timestamp_invalid_date(
    request: pytest.FixtureRequest, constructor: ConstructorEager
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark")):
        request.applymarker(pytest.mark.xfail)
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


def test_timestamp_invalid_unit_expr(constructor: ConstructorEager) -> None:
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


@given(  # type: ignore[misc]
    inputs=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
    time_unit=st.sampled_from(["ms", "us", "ns"]),
    # We keep 'ms' out for now due to an upstream bug: https://github.com/pola-rs/polars/issues/19309
    starting_time_unit=st.sampled_from(["us", "ns"]),
)
@pytest.mark.skipif(PANDAS_VERSION < (2, 2), reason="bug in old pandas")
@pytest.mark.skipif(POLARS_VERSION < (0, 20, 7), reason="bug in old Polars")
@pytest.mark.slow
def test_timestamp_hypothesis(
    inputs: datetime,
    time_unit: Literal["ms", "us", "ns"],
    starting_time_unit: Literal["ms", "us", "ns"],
) -> None:
    import polars as pl

    @nw.narwhalify
    def func(s: nw.Series) -> nw.Series:
        return s.dt.timestamp(time_unit)

    result_pl = func(pl.Series([inputs], dtype=pl.Datetime(starting_time_unit)))
    result_pd = func(pd.Series([inputs], dtype=f"datetime64[{starting_time_unit}]"))
    result_pdpa = func(
        pd.Series([inputs], dtype=f"timestamp[{starting_time_unit}][pyarrow]")
    )
    result_pa = func(pa.chunked_array([[inputs]], type=pa.timestamp(starting_time_unit)))
    assert result_pl[0] == result_pd[0]
    assert result_pl[0] == result_pdpa[0]
    assert result_pl[0] == result_pa[0].as_py()

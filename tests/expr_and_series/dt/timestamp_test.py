from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

import hypothesis.strategies as st
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    is_pyarrow_windows_no_tzdata,
)

if TYPE_CHECKING:
    from narwhals.typing import IntoSeriesT

data = {
    "a": [datetime(2021, 3, 1, 12, 34, 56, 49000), datetime(2020, 1, 2, 2, 4, 14, 715000)]
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
    constructor: Constructor,
    original_time_unit: Literal["us", "ns", "ms", "s"],
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
        request.applymarker(
            pytest.mark.xfail(reason="Backend timestamp conversion not yet implemented")
        )
    if original_time_unit == "s" and "polars" in str(constructor):
        pytest.skip("Second precision not supported in Polars")

    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (
        2,
        2,
    ):  # pragma: no cover
        pytest.skip("Requires pandas >= 2.2 for reliable pyarrow-backed timestamps")
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
    constructor: Constructor,
    original_time_unit: Literal["us", "ns", "ms", "s"],
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
        request.applymarker(
            pytest.mark.xfail(reason="Backend timestamp conversion not yet implemented")
        )
    version_conditions = [
        (
            is_pyarrow_windows_no_tzdata(constructor),
            "Timezone database is not installed on Windows",
        ),
        (
            "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2,),
            "Requires pandas >= 2.0 for pyarrow support",
        ),
        (
            "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 2),
            "Requires pandas >= 2.2 for reliable timestamps",
        ),
        (
            "dask" in str(constructor) and PANDAS_VERSION < (2, 1),
            "Requires pandas >= 2.1 for dask support",
        ),
    ]

    for condition, reason in version_conditions:
        if condition:
            pytest.skip(reason)  # pragma: no cover

    if original_time_unit == "s" and "polars" in str(constructor):
        pytest.skip()
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
    constructor: Constructor,
    time_unit: Literal["ns", "us", "ms"],
    expected: list[int | None],
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
        request.applymarker(
            pytest.mark.xfail(reason="Backend timestamp conversion not yet implemented")
        )
    unsupported_backends = (
        "pandas_constructor",
        "pandas_nullable_constructor",
        "cudf",
        "modin_constructor",
    )
    if any(x in str(constructor) for x in unsupported_backends):
        pytest.skip("Backend does not support date type")

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
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
        request.applymarker(
            pytest.mark.xfail(reason="Backend timestamp conversion not yet implemented")
        )
    if "polars" in str(constructor):
        request.applymarker(
            pytest.mark.xfail(
                reason="Invalid date handling not yet implemented in Polars"
            )
        )
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


@given(
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
    def func(s: nw.Series[IntoSeriesT]) -> nw.Series[IntoSeriesT]:
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

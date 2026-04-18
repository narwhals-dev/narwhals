from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, cast

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    import dask.dataframe as dd

data = {
    "a": [datetime(2021, 3, 1, 12, 34, 56, 49000), datetime(2020, 1, 2, 2, 4, 14, 715000)]
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
        ("weekday", [1, 4]),
    ],
)
def test_datetime_attributes(
    request: pytest.FixtureRequest,
    nw_frame_constructor: Constructor,
    attribute: str,
    expected: list[int],
) -> None:
    if (
        attribute == "date"
        and "pandas" in str(nw_frame_constructor)
        and "pyarrow" not in str(nw_frame_constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    if attribute == "date" and "cudf" in str(nw_frame_constructor):
        request.applymarker(pytest.mark.xfail)
    if attribute == "nanosecond" and "ibis" in str(nw_frame_constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(getattr(nw.col("a").dt, attribute)())
    assert_equal_data(result, {"a": expected})


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
        ("weekday", [1, 4]),
    ],
)
def test_datetime_attributes_series(
    request: pytest.FixtureRequest,
    nw_eager_constructor: ConstructorEager,
    attribute: str,
    expected: list[int],
) -> None:
    if (
        attribute == "date"
        and "pandas" in str(nw_eager_constructor)
        and "pyarrow" not in str(nw_eager_constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    if attribute == "date" and "cudf" in str(nw_eager_constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df.select(getattr(df["a"].dt, attribute)())
    assert_equal_data(result, {"a": expected})


def test_datetime_chained_attributes(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if "pandas" in str(nw_eager_constructor) and "pyarrow" not in str(
        nw_eager_constructor
    ):
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(nw_eager_constructor) and "pyarrow" not in str(
        nw_eager_constructor
    ):
        request.applymarker(pytest.mark.xfail)
    if "cudf" in str(nw_eager_constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df.select(df["a"].dt.date().dt.year())
    assert_equal_data(result, {"a": [2021, 2020]})

    result = df.select(nw.col("a").dt.date().dt.year())
    assert_equal_data(result, {"a": [2021, 2020]})


def test_to_date(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(
        x in str(nw_frame_constructor)
        for x in (
            "pandas_constructor",
            "pandas_nullable_constructor",
            "cudf",
            "modin_constructor",
        )
    ):
        request.applymarker(pytest.mark.xfail)
    dates = {"a": [datetime(2001, 1, 1), None, datetime(2001, 1, 3)]}
    if "dask" in str(nw_frame_constructor):
        df_dask = cast("dd.DataFrame", nw_frame_constructor(dates))
        df_dask = cast("dd.DataFrame", df_dask.astype({"a": "timestamp[ns][pyarrow]"}))
        df = nw.from_native(df_dask)
    else:
        df = nw.from_native(nw_frame_constructor(dates))
    result = df.select(nw.col("a").dt.date())
    assert result.collect_schema() == {"a": nw.Date}

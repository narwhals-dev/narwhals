from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import is_windows

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56, 49000),
        datetime(2020, 1, 2, 2, 4, 14, 715000),
    ],
}


@pytest.mark.parametrize(
    "fmt", ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%G-W%V-%u", "%G-W%V"]
)
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string(constructor: Any, fmt: str) -> None:
    input_frame = nw.from_native(constructor(data), eager_only=True)
    input_series = input_frame["a"]

    expected_col = [datetime.strftime(d, fmt) for d in data["a"]]

    result = input_series.dt.to_string(fmt).to_list()
    if any(x in str(constructor) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]):
        # PyArrow differs from other libraries, in that %S also shows
        # the fraction of a second.
        result = [x[: x.find(".")] if "." in x else x for x in result]
    assert result == expected_col
    result = input_frame.select(nw.col("a").dt.to_string(fmt))["a"].to_list()
    if any(x in str(constructor) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]):
        # PyArrow differs from other libraries, in that %S also shows
        # the fraction of a second.
        result = [x[: x.find(".")] if "." in x else x for x in result]
    assert result == expected_col


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (datetime(2020, 1, 9), "2020-01-09T00:00:00.000000"),
        (datetime(2020, 1, 9, 12, 34, 56), "2020-01-09T12:34:56.000000"),
        (datetime(2020, 1, 9, 12, 34, 56, 123), "2020-01-09T12:34:56.000123"),
        (datetime(2020, 1, 9, 12, 34, 56, 123456), "2020-01-09T12:34:56.123456"),
    ],
)
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_iso_local_datetime(
    constructor: Any, data: datetime, expected: str
) -> None:
    def _clean_string(result: str) -> str:
        # rstrip '0' to remove trailing zeros, as different libraries handle this differently
        # if there's then a trailing `.`, remove that too.
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return result

    df = constructor({"a": [data]})
    result = (
        nw.from_native(df, eager_only=True)["a"]
        .dt.to_string("%Y-%m-%dT%H:%M:%S.%f")
        .to_list()[0]
    )
    assert _clean_string(result) == _clean_string(expected)

    result = (
        nw.from_native(df, eager_only=True)
        .select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M:%S.%f"))["a"]
        .to_list()[0]
    )
    assert _clean_string(result) == _clean_string(expected)

    result = (
        nw.from_native(df, eager_only=True)["a"]
        .dt.to_string("%Y-%m-%dT%H:%M:%S%.f")
        .to_list()[0]
    )
    assert _clean_string(result) == _clean_string(expected)

    result = (
        nw.from_native(df, eager_only=True)
        .select(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M:%S%.f"))["a"]
        .to_list()[0]
    )
    assert _clean_string(result) == _clean_string(expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [(datetime(2020, 1, 9), "2020-01-09")],
)
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_iso_local_date(
    constructor: Any, data: datetime, expected: str
) -> None:
    df = constructor({"a": [data]})
    result = (
        nw.from_native(df, eager_only=True)["a"].dt.to_string("%Y-%m-%d").to_list()[0]
    )
    assert result == expected

    result = (
        nw.from_native(df, eager_only=True)
        .select(b=nw.col("a").dt.to_string("%Y-%m-%d"))["b"]
        .to_list()[0]
    )
    assert result == expected

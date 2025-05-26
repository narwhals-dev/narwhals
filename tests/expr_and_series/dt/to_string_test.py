from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data, is_windows

data = {
    "a": [datetime(2021, 3, 1, 12, 34, 56, 49000), datetime(2020, 1, 2, 2, 4, 14, 715000)]
}


@pytest.mark.parametrize(
    "fmt", ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%G-W%V-%u", "%G-W%V"]
)
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_series(constructor_eager: ConstructorEager, fmt: str) -> None:
    input_frame = nw.from_native(constructor_eager(data), eager_only=True)
    input_series = input_frame["a"]

    expected_col = [datetime.strftime(d, fmt) for d in data["a"]]

    result = {"a": input_series.dt.to_string(fmt)}

    if any(
        x in str(constructor_eager) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]
    ):
        # PyArrow differs from other libraries, in that %S also shows
        # the fraction of a second.
        result = {"a": input_series.dt.to_string(fmt).str.replace(r"\.\d+$", "")}

    assert_equal_data(result, {"a": expected_col})


@pytest.mark.parametrize(
    "fmt", ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%G-W%V-%u", "%G-W%V"]
)
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_expr(constructor: Constructor, fmt: str) -> None:
    input_frame = nw.from_native(constructor(data))

    expected_col = [datetime.strftime(d, fmt) for d in data["a"]]

    result = input_frame.select(nw.col("a").dt.to_string(fmt).alias("b"))
    if any(x in str(constructor) for x in ["pandas_pyarrow", "pyarrow_table", "modin"]):
        # PyArrow differs from other libraries, in that %S also shows
        # the fraction of a second.
        result = input_frame.select(
            nw.col("a").dt.to_string(fmt).str.replace(r"\.\d+$", "").alias("b")
        )
    assert_equal_data(result, {"b": expected_col})


def _clean_string(result: str) -> str:
    # rstrip '0' to remove trailing zeros, as different libraries handle this differently
    # if there's then a trailing `.`, remove that too.
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def _clean_string_expr(e: Any) -> Any:
    # Same as `_clean_string` but for Expr
    return e.str.replace_all(r"0+$", "").str.replace_all(r"\.$", "")


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
def test_dt_to_string_iso_local_datetime_series(
    constructor_eager: ConstructorEager, data: datetime, expected: str
) -> None:
    df = constructor_eager({"a": [data]})
    result = (
        nw.from_native(df, eager_only=True)["a"]
        .dt.to_string("%Y-%m-%dT%H:%M:%S.%f")
        .item(0)
    )
    assert _clean_string(str(result)) == _clean_string(expected)

    result = (
        nw.from_native(df, eager_only=True)["a"]
        .dt.to_string("%Y-%m-%dT%H:%M:%S%.f")
        .item(0)
    )
    assert _clean_string(str(result)) == _clean_string(expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (datetime(2020, 1, 9, 12, 34, 56), "2020-01-09T12:34:56.000000"),
        (datetime(2020, 1, 9, 12, 34, 56, 123), "2020-01-09T12:34:56.000123"),
        (datetime(2020, 1, 9, 12, 34, 56, 123456), "2020-01-09T12:34:56.123456"),
    ],
)
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_iso_local_datetime_expr(
    constructor: Constructor,
    data: datetime,
    expected: str,
    request: pytest.FixtureRequest,
) -> None:
    if "duckdb" in str(constructor) or "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = constructor({"a": [data]})

    result = nw.from_native(df).select(
        b=_clean_string_expr(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M:%S.%f"))
    )
    assert_equal_data(result, {"b": [_clean_string(expected)]})

    result = nw.from_native(df).select(
        b=_clean_string_expr(nw.col("a").dt.to_string("%Y-%m-%dT%H:%M:%S%.f"))
    )
    assert_equal_data(result, {"b": [_clean_string(expected)]})


@pytest.mark.parametrize(("data", "expected"), [(datetime(2020, 1, 9), "2020-01-09")])
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_iso_local_date_series(
    constructor_eager: ConstructorEager, data: datetime, expected: str
) -> None:
    df = constructor_eager({"a": [data]})
    result = nw.from_native(df, eager_only=True)["a"].dt.to_string("%Y-%m-%d").item(0)
    assert str(result) == expected


@pytest.mark.parametrize(("data", "expected"), [(datetime(2020, 1, 9), "2020-01-09")])
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_dt_to_string_iso_local_date_expr(
    constructor: Constructor, data: datetime, expected: str
) -> None:
    df = constructor({"a": [data]})
    result = nw.from_native(df).select(b=nw.col("a").dt.to_string("%Y-%m-%d"))
    assert_equal_data(result, {"b": [expected]})

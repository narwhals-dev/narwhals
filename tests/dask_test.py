"""
Dask support in Narwhals is still _very_ scant.

Start with a simple test file whilst we develop the basics.
Once we're a bit further along (say, we can at least evaluate
TPC-H Q1 with Dask), then we can integrate dask tests into
the main test suite.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

pytest.importorskip("dask")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=pytest.PytestDeprecationWarning,
    )
    pytest.importorskip("dask_expr")


if sys.version_info < (3, 9):
    pytest.skip("Dask tests require Python 3.9+", allow_module_level=True)


def test_with_columns() -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))

    df = nw.from_native(dfdd)
    result = df.with_columns(
        nw.col("a") + 1,
        (nw.col("a") + nw.col("b").mean()).alias("c"),
        d=nw.col("a"),
        e=nw.col("a") + nw.col("b"),
        f=nw.col("b") - 1,
        g=nw.col("a") - nw.col("b"),
        h=nw.col("a") * 3,
        i=nw.col("a") * nw.col("b"),
    )
    compare_dicts(
        result,
        {
            "a": [2, 3, 4],
            "b": [4, 5, 6],
            "c": [6.0, 7.0, 8.0],
            "d": [1, 2, 3],
            "e": [5, 7, 9],
            "f": [3, 4, 5],
            "g": [-3, -3, -3],
            "h": [3, 6, 9],
            "i": [4, 10, 18],
        },
    )


def test_shift() -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df = nw.from_native(dfdd)
    result = df.with_columns(nw.col("a").shift(1), nw.col("b").shift(-1))
    expected = {"a": [float("nan"), 1, 2], "b": [5, 6, float("nan")]}
    compare_dicts(result, expected)


def test_cum_sum() -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df = nw.from_native(dfdd)
    result = df.with_columns(nw.col("a", "b").cum_sum())
    expected = {"a": [1, 3, 6], "b": [4, 9, 15]}
    compare_dicts(result, expected)


def test_sum() -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df = nw.from_native(dfdd)
    result = df.with_columns((nw.col("a") + nw.col("b").sum()).alias("c"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [16, 17, 18]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("left", [True, True, True, False]),
        ("right", [False, True, True, True]),
        ("both", [True, True, True, True]),
        ("neither", [False, True, True, False]),
    ],
)
def test_is_between(closed: str, expected: list[bool]) -> None:
    import dask.dataframe as dd

    data = {
        "a": [1, 4, 2, 5],
    }
    dfdd = dd.from_pandas(pd.DataFrame(data))

    df = nw.from_native(dfdd)
    result = df.with_columns(nw.col("a").is_between(1, 5, closed=closed))
    expected_dict = {"a": expected}
    compare_dicts(result, expected_dict)


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("fda", {"a": [True, False]}),
        ("edf", {"a": [False, True]}),
        ("asd", {"a": [False, False]}),
    ],
)
def test_starts_with(prefix: str, expected: dict[str, list[bool]]) -> None:
    import dask.dataframe as dd

    data = {"a": ["fdas", "edfas"]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(nw.col("a").str.starts_with(prefix))

    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("das", {"a": [True, False]}),
        ("fas", {"a": [False, True]}),
        ("asd", {"a": [False, False]}),
    ],
)
def test_ends_with(suffix: str, expected: dict[str, list[bool]]) -> None:
    import dask.dataframe as dd

    data = {"a": ["fdas", "edfas"]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(nw.col("a").str.ends_with(suffix))

    compare_dicts(result, expected)


def test_contains() -> None:
    import dask.dataframe as dd

    data = {"pets": ["cat", "dog", "rabbit and parrot", "dove"]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)

    result = df.with_columns(
        case_insensitive_match=nw.col("pets").str.contains("(?i)parrot|Dove")
    )
    expected = {
        "pets": ["cat", "dog", "rabbit and parrot", "dove"],
        "case_insensitive_match": [False, False, True, True],
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [(1, 2, {"a": ["da", "df"]}), (-2, None, {"a": ["as", "as"]})],
)
def test_str_slice(offset: int, length: int | None, expected: Any) -> None:
    import dask.dataframe as dd

    data = {"a": ["fdas", "edfas"]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)

    result_frame = df.with_columns(nw.col("a").str.slice(offset, length))
    compare_dicts(result_frame, expected)


def test_to_datetime() -> None:
    import dask.dataframe as dd

    data = {"a": ["2020-01-01T12:34:56"]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)

    format = "%Y-%m-%dT%H:%M:%S"
    result = df.with_columns(b=nw.col("a").str.to_datetime(format=format))

    expected = {
        "a": ["2020-01-01T12:34:56"],
        "b": [datetime.strptime("2020-01-01T12:34:56", format)],  # noqa: DTZ007
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": ["foo", "bar"]}, {"a": ["FOO", "BAR"]}),
        (
            {
                "a": [
                    "special case ß",
                    "ςpecial caσe",  # noqa: RUF001
                ]
            },
            {"a": ["SPECIAL CASE ẞ", "ΣPECIAL CAΣE"]},
        ),
    ],
)
def test_str_to_uppercase(
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)

    result_frame = df.with_columns(nw.col("a").str.to_uppercase())

    compare_dicts(result_frame, expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": ["FOO", "BAR"]}, {"a": ["foo", "bar"]}),
        (
            {"a": ["SPECIAL CASE ß", "ΣPECIAL CAΣE"]},
            {
                "a": [
                    "special case ß",
                    "σpecial caσe",  # noqa: RUF001
                ]
            },
        ),
    ],
)
def test_str_to_lowercase(
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)

    result_frame = df.with_columns(nw.col("a").str.to_lowercase())
    compare_dicts(result_frame, expected)


def test_select() -> None:
    import dask.dataframe as dd

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(dd.from_pandas(pd.DataFrame(data)))
    result = df.select("a", nw.col("b") + 1, (nw.col("z") * 2).alias("z*2"))
    expected = {"a": [1, 3, 2], "b": [5, 5, 7], "z*2": [14.0, 16.0, 18.0]}
    compare_dicts(result, expected)


def test_str_only_select() -> None:
    import dask.dataframe as dd

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(dd.from_pandas(pd.DataFrame(data)))
    result = df.select("a", "b")
    expected = {"a": [1, 3, 2], "b": [4, 4, 6]}
    compare_dicts(result, expected)


def test_empty_select() -> None:
    import dask.dataframe as dd

    result = (
        nw.from_native(dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]})))
        .lazy()
        .select()
        .collect()
    )
    assert result.shape == (0, 0)


def test_dt_year() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 1), datetime(2021, 1, 1)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(year=nw.col("a").dt.year())
    expected = {"a": data["a"], "year": [2020, 2021]}
    compare_dicts(result, expected)


def test_dt_month() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 1), datetime(2021, 1, 1)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(month=nw.col("a").dt.month())
    expected = {"a": data["a"], "month": [1, 1]}
    compare_dicts(result, expected)


def test_dt_day() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 1), datetime(2021, 1, 1)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(day=nw.col("a").dt.day())
    expected = {"a": data["a"], "day": [1, 1]}
    compare_dicts(result, expected)


def test_dt_hour() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 1, 1), datetime(2021, 1, 1, 2)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(hour=nw.col("a").dt.hour())
    expected = {"a": data["a"], "hour": [1, 2]}
    compare_dicts(result, expected)


def test_dt_minute() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 1, 1, 1), datetime(2021, 1, 1, 2, 2)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(minute=nw.col("a").dt.minute())
    expected = {"a": data["a"], "minute": [1, 2]}
    compare_dicts(result, expected)


def test_dt_second() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 1, 1, 1, 1), datetime(2021, 1, 1, 2, 2, 2)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(second=nw.col("a").dt.second())
    expected = {"a": data["a"], "second": [1, 2]}
    compare_dicts(result, expected)


def test_dt_millisecond() -> None:
    import dask.dataframe as dd

    data = {
        "a": [datetime(2020, 1, 1, 1, 1, 1, 1000), datetime(2021, 1, 1, 2, 2, 2, 2000)]
    }
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(millisecond=nw.col("a").dt.millisecond())
    expected = {"a": data["a"], "millisecond": [1, 2]}
    compare_dicts(result, expected)


def test_dt_microsecond() -> None:
    import dask.dataframe as dd

    data = {
        "a": [datetime(2020, 1, 1, 1, 1, 1, 1000), datetime(2021, 1, 1, 2, 2, 2, 2000)]
    }
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(microsecond=nw.col("a").dt.microsecond())
    expected = {"a": data["a"], "microsecond": [1000, 2000]}
    compare_dicts(result, expected)


def test_dt_nanosecond() -> None:
    import dask.dataframe as dd

    data = {
        "a": [datetime(2020, 1, 1, 1, 1, 1, 1000), datetime(2021, 1, 1, 2, 2, 2, 2000)]
    }
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(nanosecond=nw.col("a").dt.nanosecond())
    expected = {"a": data["a"], "nanosecond": [1000000, 2000000]}
    compare_dicts(result, expected)


def test_dt_ordinal_day() -> None:
    import dask.dataframe as dd

    data = {"a": [datetime(2020, 1, 7), datetime(2021, 2, 1)]}
    dfdd = dd.from_pandas(pd.DataFrame(data))
    df = nw.from_native(dfdd)
    result = df.with_columns(ordinal_day=nw.col("a").dt.ordinal_day())
    expected = {"a": data["a"], "ordinal_day": [7, 32]}
    compare_dicts(result, expected)

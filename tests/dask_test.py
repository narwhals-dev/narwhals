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
from narwhals.utils import parse_version
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
    request: pytest.FixtureRequest,
    data: dict[str, list[str]],
    expected: dict[str, list[str]],
) -> None:
    import dask.dataframe as dd
    import pyarrow as pa

    if (parse_version(pa.__version__) < (12, 0, 0)) and ("ß" in data["a"][0]):
        # We are marking it xfail for these conditions above
        # since the pyarrow backend will convert
        # smaller cap 'ß' to upper cap 'ẞ' instead of 'SS'
        request.applymarker(pytest.mark.xfail)

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

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}
df_pandas = pd.DataFrame(data)
df_polars = pl.LazyFrame(data)

if os.environ.get("CI", None):
    try:
        import modin.pandas as mpd
    except ImportError:  # pragma: no cover
        df_mpd = df_pandas.copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            df_mpd = mpd.DataFrame(data)
else:  # pragma: no cover
    df_mpd = df_pandas.copy()


base_err_msg = "Anonymous expressions are not supported in "

@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_mpd],
)
def test_prefix(df_raw: Any) -> None:
    prefix = "pre_"
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("foo", "BAR").name.prefix(prefix))
    result_native = nw.to_native(result)
    expected = [prefix + k for k in data]
    assert result.columns == expected
    assert list(result_native.columns) == expected

    if not isinstance(df_raw, (pl.LazyFrame, pl.DataFrame)):
        with pytest.raises(ValueError, match=base_err_msg + "`.name.prefix`."):
            df.select(nw.all().name.prefix(prefix))


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_mpd],
)
def test_suffix(df_raw: Any) -> None:
    suffix = "_post"
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("foo", "BAR").name.suffix(suffix))
    result_native = nw.to_native(result)
    expected = [k + suffix for k in data]

    assert result.columns == expected
    assert list(result_native.columns) == expected

    if not isinstance(df_raw, (pl.LazyFrame, pl.DataFrame)):
        with pytest.raises(ValueError, match=base_err_msg + "`.name.suffix`."):
            df.select(nw.all().name.suffix(suffix))



@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_mpd],
)
def test_to_lowercase(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("foo", "BAR").name.to_lowercase())
    result_native = nw.to_native(result)
    expected = [k.lower() for k in data]

    assert result.columns == expected
    assert list(result_native.columns) == expected

    if not isinstance(df_raw, (pl.LazyFrame, pl.DataFrame)):
        with pytest.raises(ValueError, match=base_err_msg + "`.name.to_lowercase`."):
            df.select(nw.all().name.to_lowercase())



@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_mpd],
)
def test_to_uppercase(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("foo", "BAR").name.to_uppercase())
    result_native = nw.to_native(result)
    expected = [k.upper() for k in data]

    assert result.columns == expected
    assert list(result_native.columns) == expected

    if not isinstance(df_raw, (pl.LazyFrame, pl.DataFrame)):
        with pytest.raises(ValueError, match=base_err_msg + "`.name.to_uppercase`."):
            df.select(nw.all().name.to_uppercase())

@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_mpd],
)
def test_map(df_raw: Any) -> None:
    func = lambda s: s[:2].lower()  # noqa: E731

    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("foo", "BAR").name.map(func))
    result_native = nw.to_native(result)
    expected = [func(k) for k in data]

    assert result.columns == expected
    assert list(result_native.columns) == expected

    if not isinstance(df_raw, (pl.LazyFrame, pl.DataFrame)):
        with pytest.raises(ValueError, match=base_err_msg + "`.name.map`."):
            df.select(nw.all().name.map(func))

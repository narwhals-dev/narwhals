from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals.stable.v1 as nw

# Don't move this into typechecking block, for coverage
# purposes
from narwhals.typing import IntoDataFrame  # noqa: TCH001
from tests.utils import compare_dicts
from tests.utils import maybe_get_modin_df

df_pandas = pd.DataFrame({"a": ["fdas", "edfas"]})
df_lazy = pl.LazyFrame({"a": ["fdas", "edfas"]})
df_mpd = maybe_get_modin_df(df_pandas)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_mpd],
)
def test_ends_with(df_raw: IntoDataFrame) -> None:
    df = nw.from_native(df_raw).lazy()
    result = df.select(nw.col("a").str.ends_with("das"))
    result_native = nw.to_native(result)
    expected = {
        "a": [True, False],
    }
    compare_dicts(result_native, expected)
    result = df.select(df.collect()["a"].str.ends_with("das"))
    result_native = nw.to_native(result)
    expected = {
        "a": [True, False],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_mpd],
)
def test_starts_with(df_raw: Any) -> None:
    df = nw.from_native(df_raw).lazy()
    result = df.select(nw.col("a").str.starts_with("fda"))
    result_native = nw.to_native(result)
    expected = {
        "a": [True, False],
    }
    compare_dicts(result_native, expected)
    result = df.select(df.collect()["a"].str.starts_with("fda"))
    result_native = nw.to_native(result)
    expected = {
        "a": [True, False],
    }
    compare_dicts(result_native, expected)

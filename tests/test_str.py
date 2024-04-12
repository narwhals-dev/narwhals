from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

df_pandas = pd.DataFrame({"a": ["fdas", "edfas"]})
df_polars = pl.LazyFrame({"a": ["fdas", "edfas"]})

if os.environ.get("CI", None):
    import modin.pandas as mpd

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_mpd = mpd.DataFrame({"a": ["fdas", "edfas"]})
else:  # pragma: no cover
    df_mpd = df_pandas.copy()


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_mpd],
)
def test_ends_with(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
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

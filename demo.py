# ruff: noqa
# type: ignore
from typing import Any
import polars as pl
# import modin.pandas as mpd

import narwhals as nw


def func(df_raw):
    df = nw.DataFrame(df_raw)
    res = df.with_columns(
        d=nw.col("a") + 1,
        e=nw.col("a") + nw.col("b"),
    )
    res = res.group_by(["a"]).agg(
        nw.col("b").sum(),
        d=nw.col("c").sum(),
        # e=nw.len(),
    )
    return nw.to_native(res)


import pandas as pd

df = pd.DataFrame({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
print(func(df))
# df = mpd.DataFrame({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
# print(func(df))
df = pl.DataFrame({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
print(func(df))
df = pl.LazyFrame({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
print(func(df).collect())

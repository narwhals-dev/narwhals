from __future__ import annotations

import pandas as pd

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_scalar_reduction() -> None:
    import dask.dataframe as dd

    dfdd = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df = nw.from_native(dfdd)
    result = df.select(
        nw.col("a").min().alias("min"),
        nw.col("b").max().alias("max"),
        nw.col("a", "b").mean(),
    )
    expected = {"min": [1], "max": [6], "a": [2], "b": [5]}
    compare_dicts(result, expected)

    result = df.select((nw.col("a") + nw.col("b").max()).alias("x"))
    expected = {"x": [7, 8, 9]}
    compare_dicts(result, expected)

    result = df.select(nw.col("a"), nw.col("b").min())
    expected = {"a": [1, 2, 3], "b": [4, 4, 4]}
    compare_dicts(result, expected)

    result = df.select(nw.col("a").max(), nw.col("b"))
    expected = {"a": [3, 3, 3], "b": [4, 5, 6]}
    compare_dicts(result, expected)

    result = df.select(nw.col("a"), nw.col("b").min().alias("min"))

    expected = {"a": [1, 2, 3], "min": [4, 4, 4]}
    compare_dicts(result, expected)

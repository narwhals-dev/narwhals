"""
Dask support in Narwhals is still _very_ scant.

Start with a simple test file whilst we develop the basics.
Once we're a bit further along (say, we can at least evaluate
TPC-H Q1 with Dask), then we can integrate dask tests into
the main test suite.
"""

import dask.dataframe as dd
import pandas as pd

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_with_columns() -> None:
    dfdd = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))

    df = nw.from_native(dfdd)
    df = df.with_columns(
        nw.col("a") + 1,
        c=nw.col("a") + nw.col("b").mean(),
        d=nw.col("a"),
        e=nw.col("a") + nw.col("b"),
    )

    result = nw.to_native(df).compute()
    compare_dicts(
        result,
        {
            "a": [2, 3, 4],
            "b": [4, 5, 6],
            "c": [6.0, 7.0, 8.0],
            "d": [1, 2, 3],
            "e": [5, 7, 9],
        },
    )

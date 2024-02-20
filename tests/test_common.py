# ruff: noqa
import pandas as pd

import narwhals


def test_common():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dfx, plx = narwhals.to_polars_api(df, api_version="0.20")

    result = dfx.with_columns(
        c=plx.col("a") + plx.col("b"),
        d=plx.col("a") - plx.col("a").mean(),
    )
    print(result.dataframe)
    result = dfx.with_columns(plx.all() * 2)
    print(result.dataframe)

    result = dfx.with_columns(
        horizonal_sum=plx.sum_horizontal(plx.col("a"), plx.col("b"))
    )
    print(result.dataframe)
    result = dfx.select(plx.all().sum())
    print(result.dataframe)

import pandas as pd
import polars as pl

import narwhals as nw


def test_cast_253() -> None:
    df_polars = pl.DataFrame({"a": [1]})
    result = nw.from_native(df_polars, eager_only=True).select(
        nw.col("a").cast(nw.String) + "hi"
    )["a"][0]
    assert result == "1hi"

    df_pandas = pd.DataFrame({"a": [1]})
    result = nw.from_native(df_pandas, eager_only=True).select(
        nw.col("a").cast(nw.String) + "hi"
    )["a"][0]
    assert result == "1hi"

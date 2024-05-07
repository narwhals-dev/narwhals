from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

df_pandas = pd.DataFrame({"a": ["2020-01-01T12:34:56"]})
df_polars = pl.DataFrame({"a": ["2020-01-01T12:34:56"]})


@pytest.mark.parametrize("df_any", [df_pandas, df_polars])
def test_to_datetime(df_any: Any) -> None:
    result = nw.from_native(df_any, eager_only=True).select(
        b=nw.col("a").str.to_datetime(format="%Y-%m-%dT%H:%M:%S")
    )["b"][0]
    assert result == datetime(2020, 1, 1, 12, 34, 56)

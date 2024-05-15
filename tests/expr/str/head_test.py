from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_str_head(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.col("a").str.head(3))
    expected = {
        "a": ["foo", "bar"],
    }
    compare_dicts(result, expected)
    result = df.select(df["a"].str.head(3))
    compare_dicts(result, expected)

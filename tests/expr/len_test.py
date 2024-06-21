from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {"a": list("xyz"), "b": [1, 2, 1]}
expected = {"a1": [2], "a2": [1]}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_len(constructor: Any) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw).select(
        nw.col("a").filter(nw.col("b") == 1).len().alias("a1"),
        nw.col("a").filter(nw.col("b") == 2).len().alias("a2"),
    )

    compare_dicts(nw.to_native(df), expected)

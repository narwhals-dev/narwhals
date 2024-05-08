from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_over_single(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.with_columns(nw.all().cum_sum())
    expected = {
        "a": [0, 1, 3, 6, 10],
        "b": [1, 3, 6, 11, 14],
        "c": [5, 9, 12, 14, 15],
    }
    compare_dicts(result, expected)
    result = df.select(
        df["a"].cum_sum(),
        df["b"].cum_sum(),
        df["c"].cum_sum(),
    )
    compare_dicts(result, expected)

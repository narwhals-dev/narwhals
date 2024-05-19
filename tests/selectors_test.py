from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from narwhals.selectors import by_dtype
from tests.utils import compare_dicts

data = {"a": [1, 1, 2], "b": ["a", "b", "c"], "c": [4.0, 5.0, 6.0]}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_selecctors(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(by_dtype([nw.Int64, nw.Float64]) + 1))
    expected = {"a": [2, 2, 3], "c": [5.0, 6.0, 7.0]}
    compare_dicts(result, expected)

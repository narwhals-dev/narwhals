from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2, 3, 2],
    "b": [1, 2, 3, 2, 1],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_is_last_distinct(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().is_last_distinct())
    expected = {
        "a": [False, True, False, True, True],
        "b": [False, False, True, True, True],
    }
    compare_dicts(result, expected)

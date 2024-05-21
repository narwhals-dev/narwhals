from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, None, None, 3.0],
    "b": [1.0, None, 4, 5.0],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_null_count(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().null_count())
    expected = {
        "a": [2],
        "b": [1],
    }
    compare_dicts(result, expected)

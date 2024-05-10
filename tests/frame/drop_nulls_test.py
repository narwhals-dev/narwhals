from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_drop_nulls(constructor: Any) -> None:
    result = nw.from_native(constructor(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": [3.0, 5.0],
    }
    compare_dicts(result, expected)
    result = nw.from_native(constructor(data)).lazy().drop_nulls()
    compare_dicts(result, expected)

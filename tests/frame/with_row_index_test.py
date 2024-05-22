from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_with_row_index(constructor: Any) -> None:
    result = nw.from_native(constructor(data)).with_row_index()
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "index": [0, 1]}
    compare_dicts(result, expected)
    result = nw.from_native(constructor(data)).lazy().with_row_index("foo")
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "foo": [0, 1]}
    compare_dicts(result, expected)

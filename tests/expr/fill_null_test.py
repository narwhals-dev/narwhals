from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [0.0, None, 2, 3, 4],
    "b": [1.0, None, None, 5, 3],
    "c": [5.0, None, 3, 2, 1],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_over_single(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.with_columns(nw.all().fill_null(99))
    expected = {
        "a": [0.0, 99, 2, 3, 4],
        "b": [1.0, 99, 99, 5, 3],
        "c": [5.0, 99, 3, 2, 1],
    }
    compare_dicts(result, expected)
    result = df.with_columns(
        a=df["a"].fill_null(99),
        b=df["b"].fill_null(99),
        c=df["c"].fill_null(99),
    )
    compare_dicts(result, expected)

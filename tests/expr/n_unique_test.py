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
def test_over_single(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().n_unique())
    expected = {
        "a": [3],
        "b": [4],
    }
    compare_dicts(result, expected)
    assert df["a"].n_unique() == 3
    assert df["b"].n_unique() == 4
    compare_dicts(result, expected)

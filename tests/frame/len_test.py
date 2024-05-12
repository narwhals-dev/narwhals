from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_drop_nulls(constructor: Any) -> None:
    result = len(nw.from_native(constructor(data)))
    assert result == 4

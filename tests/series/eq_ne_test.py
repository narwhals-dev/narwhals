from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

data = [1, 2, 3]


@pytest.mark.parametrize("constructor", [pd.Series, pl.Series])
def test_eq_ne(constructor: Any) -> None:
    s = nw.from_native(constructor(data), series_only=True)
    assert (s == 1).to_numpy().tolist() == [True, False, False]
    assert (s != 1).to_numpy().tolist() == [False, True, True]

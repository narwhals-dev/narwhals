from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

data = [1, 2, 3]


@pytest.mark.parametrize("constructor", [pd.Series, pl.Series])
def test_to_list(constructor: Any) -> None:
    s = nw.from_native(constructor(data), series_only=True)
    assert s.to_list() == [1, 2, 3]

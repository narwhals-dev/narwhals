from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_dt_year(constructor: Any) -> None:
    df = nw.LazyFrame(constructor(data))
    result = nw.to_native(df.select(nw.col("a").dt.year()))
    expected = {"a": [2020, 2020, 2020]}
    compare_dicts(result, expected)
    result = nw.to_native(df.select(df.collect()["a"].dt.year()))
    expected = {"a": [2020, 2020, 2020]}
    compare_dicts(result, expected)

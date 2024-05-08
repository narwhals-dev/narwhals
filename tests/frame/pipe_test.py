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
def test_str_head(constructor: Any) -> None:
    result = nw.from_native(constructor(data)).pipe(
        lambda _df: _df.select([x for x in _df.columns if len(x) == 2])
    )
    expected = {"ab": ["foo", "bars"]}
    compare_dicts(result, expected)
    result = (
        nw.from_native(constructor(data))
        .lazy()
        .pipe(lambda _df: _df.select([x for x in _df.columns if len(x) == 2]))
    )
    expected = {"ab": ["foo", "bars"]}
    compare_dicts(result, expected)

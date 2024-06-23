from typing import Any

import pandas as pd
import polars as pl
import pytest

from tests.utils import compare_dicts
from tests.utils import nw

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_pipe(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    columns = df.columns
    result = df.pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    expected = {"ab": ["foo", "bars"]}
    compare_dicts(result, expected)
    result = df.lazy().pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    expected = {"ab": ["foo", "bars"]}
    compare_dicts(result, expected)

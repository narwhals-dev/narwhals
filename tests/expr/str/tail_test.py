from typing import Any

import pandas as pd
import polars as pl
import pytest

from tests.utils import compare_dicts
from tests.utils import nw

data = {
    "a": ["foo", "bars"],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_str_tail(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    expected = {
        "a": ["foo", "ars"],
    }

    result_frame = df.select(nw.col("a").str.tail(3))
    compare_dicts(result_frame, expected)

    result_series = df["a"].str.tail(3)
    assert result_series.to_numpy().tolist() == expected["a"]

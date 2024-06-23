from typing import Any

import pandas as pd
import polars as pl
import pytest

from tests.utils import compare_dicts
from tests.utils import nw

data = {
    "a": [1, 1, 2],
    "b": [1, 2, 3],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_is_unique(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().is_unique())
    expected = {
        "a": [False, False, True],
        "b": [True, True, True],
    }
    compare_dicts(result, expected)

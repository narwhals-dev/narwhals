from typing import Any

import numpy as np
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
def test_with_columns(constructor: Any) -> None:
    result = nw.from_native(constructor(data)).with_columns(d=np.array([4, 5]))
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "d": [4, 5]}
    compare_dicts(result, expected)

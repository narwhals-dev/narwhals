from typing import Any

import pandas as pd
import polars as pl
import pytest

from tests.utils import compare_dicts
from tests.utils import nw

data = [1, 2, 3]


@pytest.mark.parametrize("constructor", [pd.Series, pl.Series])
def test_eq_ne(constructor: Any) -> None:
    df = nw.from_native(constructor(data), series_only=True).alias("").to_frame()
    compare_dicts(df, {"": [1, 2, 3]})

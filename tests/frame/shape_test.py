from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals as nw


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame, pa.table])
def test_shape(constructor: Any) -> None:
    result = nw.DataFrame(constructor({"a": [1, 2], "b": [4, 5], "c": [7, 8]})).shape
    expected = (2, 3)
    assert result == expected

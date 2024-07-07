from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw


@pytest.mark.parametrize(
    "constructor", [pd.Series, pl.Series, lambda x: pa.table({"a": x})["a"]]
)
def test_shape(constructor: Any) -> None:
    result = nw.from_native(constructor([1, 2]), series_only=True).shape
    expected = (2,)
    assert result == expected

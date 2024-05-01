import pandas as pd
import polars as pl
import pytest

import narwhals as nw


def test_native_vs_non_native() -> None:
    s = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(TypeError, match="Perhaps you forgot"):
        nw.from_native(df).filter(s > 1)
    s = pl.Series([1, 2, 3])
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(TypeError, match="Perhaps you forgot"):
        nw.from_native(df).filter(s > 1)

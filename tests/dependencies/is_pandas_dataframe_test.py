from __future__ import annotations

import pandas as pd
import pytest

from narwhals.stable.v1.dependencies import is_pandas_dataframe


def test_is_pandas_dataframe() -> None:
    assert is_pandas_dataframe(pd.DataFrame())


def test_not_is_pandas_dataframe() -> None:
    pytest.importorskip("polars")
    import polars as pl

    assert not is_pandas_dataframe(pl.DataFrame())

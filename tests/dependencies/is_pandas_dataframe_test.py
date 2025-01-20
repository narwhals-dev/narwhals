from __future__ import annotations

import pytest

from narwhals.stable.v1.dependencies import is_pandas_dataframe


def test_is_pandas_dataframe() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    import pandas as pd
    import polars as pl

    assert is_pandas_dataframe(pd.DataFrame())
    assert not is_pandas_dataframe(pl.DataFrame())

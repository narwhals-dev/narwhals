from __future__ import annotations

from datetime import datetime

import hypothesis.strategies as st
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION


@given(dates=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)))  # type: ignore[misc]
@pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0),
    reason="pyarrow dtype not available",
)
@pytest.mark.slow
def test_ordinal_day(dates: datetime) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    import pandas as pd
    import polars as pl

    result_pd = nw.from_native(pd.Series([dates]), series_only=True).dt.ordinal_day()[0]
    result_pdms = nw.from_native(
        pd.Series([dates]).dt.as_unit("ms"), series_only=True
    ).dt.ordinal_day()[0]
    result_pda = nw.from_native(
        pd.Series([dates]).convert_dtypes(dtype_backend="pyarrow"), series_only=True
    ).dt.ordinal_day()[0]
    result_pdn = nw.from_native(
        pd.Series([dates]).convert_dtypes(dtype_backend="numpy_nullable"),
        series_only=True,
    ).dt.ordinal_day()[0]
    result_pl = nw.from_native(pl.Series([dates]), series_only=True).dt.ordinal_day()[0]
    assert result_pd == result_pl
    assert result_pda == result_pl
    assert result_pdn == result_pl
    assert result_pdms == result_pl

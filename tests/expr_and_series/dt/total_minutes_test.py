from __future__ import annotations

from datetime import timedelta

import hypothesis.strategies as st
import pandas as pd
import polars as pl
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw


@given(
    timedeltas=st.timedeltas(
        min_value=-timedelta(days=5, minutes=70, seconds=10),
        max_value=timedelta(days=3, minutes=90, seconds=60),
    )
)  # type: ignore[misc]
@pytest.mark.slow
def test_total_minutes(
    timedeltas: timedelta,
    request: pytest.FixtureRequest,
    pandas_version: tuple[int, ...],
) -> None:
    if pandas_version < (2, 2, 0):
        request.applymarker(pytest.mark.skipif(reason="pyarrow dtype not available"))
    result_pd = nw.from_native(
        pd.Series([timedeltas]), series_only=True
    ).dt.total_minutes()[0]
    result_pdns = nw.from_native(
        pd.Series([timedeltas]).dt.as_unit("ns"), series_only=True
    ).dt.total_minutes()[0]
    result_pda = nw.from_native(
        pd.Series([timedeltas]).convert_dtypes(dtype_backend="pyarrow"), series_only=True
    ).dt.total_minutes()[0]
    result_pdn = nw.from_native(
        pd.Series([timedeltas]).convert_dtypes(dtype_backend="numpy_nullable"),
        series_only=True,
    ).dt.total_minutes()[0]
    result_pl = nw.from_native(
        pl.Series([timedeltas]), series_only=True
    ).dt.total_minutes()[0]
    assert result_pd == result_pl
    assert result_pda == result_pl
    assert result_pdn == result_pl
    assert result_pdns == result_pl

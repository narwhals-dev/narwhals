from __future__ import annotations

from datetime import timedelta

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION


@given(
    timedeltas=st.timedeltas(
        min_value=-timedelta(days=5, minutes=70, seconds=10),
        max_value=timedelta(days=3, minutes=90, seconds=60),
    )
)  # type: ignore[misc]
@pytest.mark.skipif(
    PANDAS_VERSION < (2, 2, 0),
    reason="pyarrow dtype not available",
)
@pytest.mark.slow
def test_total_minutes(timedeltas: timedelta) -> None:
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
    assert result_pda == result_pd
    assert result_pdn == result_pd
    assert result_pdns == result_pd

    try:
        import polars as pl
    except ImportError:
        pass
    else:
        result_pl = nw.from_native(
            pl.Series([timedeltas]), series_only=True
        ).dt.total_minutes()[0]
        assert result_pl == result_pd

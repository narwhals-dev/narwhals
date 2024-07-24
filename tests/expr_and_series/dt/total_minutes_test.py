from __future__ import annotations

from datetime import datetime
from datetime import timedelta

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56, 49000),
        datetime(2020, 1, 2, 2, 4, 14, 715000),
    ],
}
data_timedelta = {
    "a": [
        None,
        timedelta(minutes=1, seconds=1, milliseconds=1, microseconds=1),
    ],
    "b": [
        timedelta(milliseconds=2),
        timedelta(milliseconds=1, microseconds=300),
    ],
    "c": np.array([None, 20], dtype="timedelta64[ns]"),
}


@given(
    timedeltas=st.timedeltas(
        min_value=-timedelta(days=5, minutes=70, seconds=10),
        max_value=timedelta(days=3, minutes=90, seconds=60),
    )
)  # type: ignore[misc]
@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.2.0"),
    reason="pyarrow dtype not available",
)
@pytest.mark.slow()
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
    result_pl = nw.from_native(
        pl.Series([timedeltas]), series_only=True
    ).dt.total_minutes()[0]
    assert result_pd == result_pl
    assert result_pda == result_pl
    assert result_pdn == result_pl
    assert result_pdns == result_pl

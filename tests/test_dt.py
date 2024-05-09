from __future__ import annotations

from datetime import datetime
from typing import Any

import hypothesis.strategies as st
import pandas as pd
import polars as pl
import pytest
from hypothesis import given

import narwhals as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56),
        datetime(2020, 1, 2, 2, 4, 14),
    ],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("year", [2021, 2020]),
        ("month", [3, 1]),
        ("day", [1, 2]),
        ("hour", [12, 2]),
        ("minute", [34, 4]),
        ("second", [56, 14]),
        ("ordinal_day", [60, 2]),
    ],
)
def test_dt_year(attribute: str, expected: list[int], constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = nw.to_native(df.select(getattr(nw.col("a").dt, attribute)()))
    compare_dicts(result, {"a": expected})
    result = nw.to_native(df.select(getattr(df["a"].dt, attribute)()))
    compare_dicts(result, {"a": expected})


@given(dates=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)))  # type: ignore[misc]
@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"),
    reason="pyarrow dtype not available",
)
def test_ordinal_day(dates: datetime) -> None:
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

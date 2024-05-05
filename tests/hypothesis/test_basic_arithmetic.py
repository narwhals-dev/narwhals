from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import narwhals as nw


@given(
    st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
    st.lists(
        st.floats(min_value=-9223372036854775807.0, max_value=9223372036854775807.0),
        min_size=3,
        max_size=3,
    ),
)  # type: ignore[misc]
@pytest.mark.slow()
def test_mean(
    integer: st.SearchStrategy[list[int]],
    floats: st.SearchStrategy[float],
) -> None:
    df_pandas = pd.DataFrame(
        {
            "integer": integer,
            "floats": floats,
        }
    )
    df_polars = pl.DataFrame(
        {
            "integer": integer,
            "floats": floats,
        },
    )
    df_nw1 = nw.DataFrame(df_pandas)
    df_nw2 = nw.DataFrame(df_polars)

    assert_allclose(
        nw.to_native(df_nw1.select(nw.col("integer").mean())),
        nw.to_native(df_nw2.select(nw.col("integer").mean())),
    )
    assert_allclose(
        nw.to_native(df_nw1.select(nw.col("floats").mean())),
        nw.to_native(df_nw2.select(nw.col("floats").mean())),
    )

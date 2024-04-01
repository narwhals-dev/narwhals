from __future__ import annotations

import pandas as pd
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

import narwhals as nw


@given(
    st.lists(st.integers(), min_size=3, max_size=3),
    st.lists(st.datetimes(), min_size=3, max_size=3),
    st.lists(st.floats(), min_size=3, max_size=3),
    st.lists(st.text(min_size=1), min_size=3, max_size=3),
) 
def test_isnull(
    integers: st.SearchStrategy,
    datetimes: st.SearchStrategy,
    floats: st.SearchStrategy,
    text: st.SearchStrategy,
) -> None:
    dfpd = pd.DataFrame(
        {
            "integer": integers,
            "date": datetimes,
            "floats": floats,
            "string": text,
        }
    )
    dfpl = pl.DataFrame(
        {
            "integer": integers,
            "date": datetimes,
            "floats": floats,
            "string": text,
        }
    )
    df_nw1 = nw.DataFrame(dfpd)
    df_nw2 = nw.DataFrame(dfpl)

    assert df_nw1.select(nw.col("integer").is_null()) == df_nw2.select(
        nw.col("integer").is_null()
    )
    assert df_nw1.select(nw.col("date").is_null()) == df_nw2.select(
        nw.col("date").is_null()
    )
    assert df_nw1.select(nw.col("floats").is_null()) == df_nw2.select(
        nw.col("floats").is_null()
    )
    assert df_nw1.select(nw.col("strings").is_null()) == df_nw2.select(
        nw.col("strings").is_null()
    )

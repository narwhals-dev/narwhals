from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

import narwhals as nw

from hypothesis import given, settings, strategies as st


@given(st.lists(st.integers(), min_size=3, max_size=3),
       st.lists(st.datetimes(), min_size=3, max_size=3),
       st.lists(st.floats(), min_size=3, max_size=3),
       st.lists(st.text(min_size=1), min_size=3, max_size=3),
)
def test_isnull(integers, datetimes, floats, text):
    
    dfpd = pd.DataFrame({"integer": integers,
                         "date": datetimes,
                         "floats": floats,
                         "string": text,
                         })
    dfpl = pl.DataFrame({"integer": integers,
                         "date": datetimes,
                         "floats": floats,
                         "string": text,
                         })
    df_nw1 = nw.DataFrame(dfpd)
    df_nw2 = nw.DataFrame(dfpl)
    
    assert (df_nw1.select(nw.col("integer").is_null())
            == 
            df_nw2.select(nw.col("integer").is_null())
    )
    assert (df_nw1.select(nw.col("date").is_null())
            == 
            df_nw2.select(nw.col("date").is_null())
    )
    assert (df_nw1.select(nw.col("floats").is_null())
            == 
            df_nw2.select(nw.col("floats").is_null())
    )
    assert (df_nw1.select(nw.col("strings").is_null())
            == 
            df_nw2.select(nw.col("strings").is_null())
    )
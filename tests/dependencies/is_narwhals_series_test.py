from __future__ import annotations

import pandas as pd
import polars as pl

import narwhals as nw
import narwhals.stable.v1 as nws
from narwhals.stable.v1.dependencies import is_narwhals_series


def test_is_narwhals_series() -> None:
    s = [1, 2, 3]
    s_pd = pd.Series(s)
    s_pl = pl.Series(s)

    assert is_narwhals_series(nw.from_native(s_pd, series_only=True))
    assert is_narwhals_series(nws.from_native(s_pd, series_only=True))
    assert is_narwhals_series(nw.from_native(s_pl, series_only=True))
    assert is_narwhals_series(nws.from_native(s_pl, series_only=True))
    assert not is_narwhals_series(s_pd)
    assert not is_narwhals_series(s_pl)

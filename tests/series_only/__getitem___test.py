from __future__ import annotations

import polars as pl
import pytest

import narwhals.stable.v1 as nw


def test_getitem() -> None:
    spl = pl.Series([1, 2, 3])
    assert spl[spl[0, 1]].equals(pl.Series([2, 3]))

    snw = nw.from_native(spl, series_only=True)
    assert snw[snw[0, 1]].to_native().equals(pl.Series([2, 3]))

    spl = pl.Series([1, 2, 3])
    snw = nw.from_native(spl, series_only=True)
    assert pytest.raises(TypeError, lambda: snw[snw[True, False]])

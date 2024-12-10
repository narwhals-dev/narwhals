from __future__ import annotations

import polars as pl

import narwhals as nw
import narwhals.stable.v1 as nws
from narwhals.stable.v1.dependencies import is_narwhals_lazyframe


def test_is_narwhals_lazyframe() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    assert is_narwhals_lazyframe(nw.from_native(lf))
    assert is_narwhals_lazyframe(nws.from_native(lf))
    assert not is_narwhals_lazyframe(lf)

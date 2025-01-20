from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw


def test_native_namespace_pl() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pl
    assert nw.get_native_namespace(df.to_native()) is pl
    assert nw.get_native_namespace(df.lazy().to_native()) is pl
    assert nw.get_native_namespace(df["a"].to_native()) is pl


def test_native_namespace_pd() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pd
    assert nw.get_native_namespace(df.to_native()) is pd
    assert nw.get_native_namespace(df["a"].to_native()) is pd


def test_native_namespace_pa() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw.from_native(pa.table({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pa
    assert nw.get_native_namespace(df.to_native()) is pa
    assert nw.get_native_namespace(df["a"].to_native()) is pa


def test_get_native_namespace_invalid() -> None:
    with pytest.raises(TypeError, match="Could not get native namespace"):
        nw.get_native_namespace(1)

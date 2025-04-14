from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import Frame


def test_native_namespace_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df: Frame = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pl
    assert nw.get_native_namespace(df.to_native()) is pl
    assert nw.get_native_namespace(df.lazy().to_native()) is pl
    assert nw.get_native_namespace(df["a"].to_native()) is pl
    assert nw.get_native_namespace(df, df["a"].to_native()) is pl


def test_native_namespace_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df: Frame = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    assert nw.get_native_namespace(df) is pd
    assert nw.get_native_namespace(df.to_native()) is pd
    assert nw.get_native_namespace(df["a"].to_native()) is pd
    assert nw.get_native_namespace(df, df["a"].to_native()) is pd


def test_native_namespace_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df: Frame = nw.from_native(pa.table({"a": [1, 2, 3]}), eager_only=True)
    assert nw.get_native_namespace(df) is pa
    assert nw.get_native_namespace(df.to_native()) is pa
    assert nw.get_native_namespace(df, df["a"].to_native()) is pa


def test_get_native_namespace_invalid() -> None:
    with pytest.raises(TypeError, match="Could not get native namespace"):
        nw.get_native_namespace(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least one object"):
        nw.get_native_namespace()


def test_get_native_namespace_invalid_cross() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    import pandas as pd
    import polars as pl

    with pytest.raises(ValueError, match="Found objects with different"):
        nw.get_native_namespace(pd.Series([1]), pl.Series([2]))

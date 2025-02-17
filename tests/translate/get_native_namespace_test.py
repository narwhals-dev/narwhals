from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v1 as nw
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

if TYPE_CHECKING:
    from narwhals.typing import Frame


def test_native_namespace() -> None:
    df: Frame = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pl
    assert nw.get_native_namespace(df.to_native()) is pl
    assert nw.get_native_namespace(df.lazy().to_native()) is pl
    assert nw.get_native_namespace(df["a"].to_native()) is pl
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    assert nw.get_native_namespace(df) is pd
    assert nw.get_native_namespace(df.to_native()) is pd
    assert nw.get_native_namespace(df["a"].to_native()) is pd
    df = nw.from_native(pa.table({"a": [1, 2, 3]}), eager_only=True)
    assert nw.get_native_namespace(df) is pa
    assert nw.get_native_namespace(df.to_native()) is pa
    assert nw.get_native_namespace(df["a"].to_native()) is pa


def test_get_native_namespace_invalid() -> None:
    with pytest.raises(TypeError, match="Could not get native namespace"):
        nw.get_native_namespace(1)  # type: ignore[arg-type]

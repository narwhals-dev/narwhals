from __future__ import annotations

import sys

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PYARROW_VERSION
from tests.utils import assert_equal_data


@pytest.mark.xfail(PYARROW_VERSION < (14,), reason="too old")
def test_from_arrow_to_arrow() -> None:
    pl = pytest.importorskip("polars")
    df = nw.from_native(pl.DataFrame({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_arrow(df, native_namespace=pa)
    assert isinstance(result.to_native(), pa.Table)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)


@pytest.mark.xfail(PYARROW_VERSION < (14,), reason="too old")
def test_from_arrow_to_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    pl = pytest.importorskip("polars")
    tbl = pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]})
    monkeypatch.delitem(sys.modules, "pandas")
    df = nw.from_native(tbl, eager_only=True)
    result = nw.from_arrow(df, native_namespace=pl)
    assert isinstance(result.to_native(), pl.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)
    assert "pandas" not in sys.modules


@pytest.mark.xfail(PYARROW_VERSION < (14,), reason="too old")
def test_from_arrow_to_pandas() -> None:
    df = nw.from_native(pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_arrow(df, native_namespace=pd)
    assert isinstance(result.to_native(), pd.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_arrow_invalid() -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        nw.from_arrow({"a": [1]}, native_namespace=pa)  # type: ignore[arg-type]

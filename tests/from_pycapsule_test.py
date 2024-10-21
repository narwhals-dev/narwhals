from __future__ import annotations

import sys

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import assert_equal_data


def test_from_arrow_to_arrow(
    request: pytest.FixtureRequest, pyarrow_version: tuple[int, ...]
) -> None:
    if pyarrow_version < (14,):
        request.applymarker(pytest.mark.xfail(reason="too old"))
    df = nw.from_native(pl.DataFrame({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_arrow(df, native_namespace=pa)
    assert isinstance(result.to_native(), pa.Table)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_arrow_to_polars(
    request: pytest.FixtureRequest,
    pyarrow_version: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if pyarrow_version < (14,):
        request.applymarker(pytest.mark.xfail(reason="too old"))
    tbl = pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]})
    monkeypatch.delitem(sys.modules, "pandas")
    df = nw.from_native(tbl, eager_only=True)
    result = nw.from_arrow(df, native_namespace=pl)
    assert isinstance(result.to_native(), pl.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)
    assert "pandas" not in sys.modules


def test_from_arrow_to_pandas(
    request: pytest.FixtureRequest, pyarrow_version: tuple[int, ...]
) -> None:
    if pyarrow_version < (14,):
        request.applymarker(pytest.mark.xfail(reason="too old"))
    df = nw.from_native(pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_arrow(df, native_namespace=pd)
    assert isinstance(result.to_native(), pd.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_arrow_invalid() -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        nw.from_arrow({"a": [1]}, native_namespace=pa)  # type: ignore[arg-type]

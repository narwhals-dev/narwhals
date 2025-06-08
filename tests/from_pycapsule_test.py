from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import PYARROW_VERSION, assert_equal_data

pytest.importorskip("pyarrow")
import pyarrow as pa

from narwhals._utils import Implementation, Version


@dataclass
class FullContext:
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version


@pytest.mark.xfail(PYARROW_VERSION < (14,), reason="too old")
def test_from_arrow_to_arrow() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_arrow(df, backend=pa)
    assert isinstance(result.to_native(), pa.Table)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_arrow_to_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("polars")
    import polars as pl

    tbl = pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]})
    monkeypatch.delitem(sys.modules, "pandas")
    if PYARROW_VERSION < (14,):  # pragma: no cover
        result = nw.from_arrow(tbl, backend=pl)
    else:
        df = nw.from_native(tbl, eager_only=True)
        result = nw.from_arrow(df, backend=pl)
    assert isinstance(result.to_native(), pl.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)
    assert "pandas" not in sys.modules


@pytest.mark.xfail(PYARROW_VERSION < (14,), reason="too old")
def test_from_arrow_to_pandas() -> None:
    df = nw.from_native(pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_arrow(df, backend=pd)
    assert isinstance(result.to_native(), pd.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_arrow_invalid() -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        nw.from_arrow({"a": [1]}, backend=pa)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "context", [FullContext(Implementation.POLARS, (1, 0, 0), Version.V1)]
)
def test_from_arrow_pre_14(context: FullContext) -> None:
    pytest.importorskip("polars")
    from narwhals._polars.dataframe import PolarsDataFrame

    expected: dict[str, Any] = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    tbl = pa.table(expected)
    compliant = PolarsDataFrame.from_arrow(tbl, context=context)
    result = nw.from_native(compliant.to_polars(), eager_only=True)
    assert_equal_data(result, expected)

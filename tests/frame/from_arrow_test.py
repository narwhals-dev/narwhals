from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa

import narwhals as nw
from narwhals._utils import Implementation
from tests.utils import PYARROW_VERSION, assert_equal_data

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed


@pytest.fixture
def data() -> dict[str, Any]:
    return {"ab": [1, 2, 3], "ba": ["four", "five", None]}


@pytest.fixture
def table(data: dict[str, Any]) -> pa.Table:
    return pa.table(data)


def is_native(native: Any, backend: EagerAllowed) -> bool:
    if backend in {Implementation.PYARROW, "pyarrow"}:
        return isinstance(native, pa.Table)
    if backend in {Implementation.POLARS, "polars"}:
        import polars as pl

        return isinstance(native, pl.DataFrame)
    if backend in {Implementation.PANDAS, "pandas"}:
        import pandas as pd

        return isinstance(native, pd.DataFrame)
    msg = f"Unexpected backend {backend!r}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def test_dataframe_from_arrow_table(
    eager_backend: EagerAllowed, table: pa.Table, data: dict[str, Any]
) -> None:
    # NOTE: PyCapsule support requires `pyarrow>=14`, but this path should work in all cases
    result = nw.DataFrame.from_arrow(table, backend=eager_backend)
    assert_equal_data(result, data)
    assert is_native(result.to_native(), eager_backend)


@pytest.mark.xfail(PYARROW_VERSION < (14,), reason="too old")
def test_dataframe_from_arrow_pycapsule(
    eager_backend: EagerAllowed, table: pa.Table, data: dict[str, Any]
) -> None:
    result = nw.DataFrame.from_arrow(table, backend=eager_backend)
    supports_arrow_c_stream = nw.from_native(table)
    result = nw.DataFrame.from_arrow(supports_arrow_c_stream, backend=eager_backend)
    assert_equal_data(result, data)
    assert is_native(result.to_native(), eager_backend)


def test_dataframe_from_arrow_to_polars_no_pandas(
    monkeypatch: pytest.MonkeyPatch, table: pa.Table, data: dict[str, Any]
) -> None:
    pytest.importorskip("polars")
    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    if PYARROW_VERSION < (14,):  # pragma: no cover
        result = nw.DataFrame.from_arrow(table, backend="polars")
    else:
        supports_arrow_c_stream = nw.from_native(table)
        result = nw.DataFrame.from_arrow(supports_arrow_c_stream, backend="polars")
    assert is_native(result.to_native(), "polars")
    assert_equal_data(result, data)
    assert "pandas" not in sys.modules


def test_dataframe_from_arrow_modin(table: pa.Table, data: dict[str, Any]) -> None:
    pytest.importorskip("modin.pandas")
    result = nw.DataFrame.from_arrow(table, backend="modin")
    assert result.implementation.is_modin()
    assert_equal_data(result, data)


def test_dataframe_from_arrow_invalid(table: pa.Table, data: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        nw.DataFrame.from_arrow(data, backend=pa)  # type: ignore[arg-type]
    pytest.importorskip("sqlframe")
    with pytest.raises(ValueError, match="lazy"):
        nw.DataFrame.from_arrow(table, backend="sqlframe")  # type: ignore[arg-type]

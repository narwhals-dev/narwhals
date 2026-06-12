from __future__ import annotations

import pytest

import narwhals as nw

pytest.importorskip(
    "polars", minversion="1.3.0", reason="Too old for pycapsule in Polars"
)
pytest.importorskip(
    "pyarrow", minversion="16.0.0", reason="Too old for pycapsule in PyArrow"
)
pytest.importorskip("pyarrow.compute")

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc


def test_arrow_c_stream_test() -> None:
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    result = pa.table(df)
    expected = pa.table({"a": [1, 2, 3]})
    assert pc.all(pc.equal(result["a"], expected["a"])).as_py()


def test_arrow_c_stream_test_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    # "poison" the dunder method to make sure it actually got called above
    monkeypatch.setattr(
        "narwhals.dataframe.DataFrame.__arrow_c_stream__", lambda *_: 1 / 0
    )
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        pa.table(df)


def test_arrow_c_stream_test_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas", reason="Require pandas for `backend='pandas'`")
    # Check that fallback to PyArrow works
    monkeypatch.delattr("polars.DataFrame.__arrow_c_stream__")
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    result = pa.table(df)
    expected = pa.table({"a": [1, 2, 3]})
    assert pc.all(pc.equal(result["a"], expected["a"])).as_py()
    result_2 = nw.from_arrow(result, backend="pandas").to_arrow()
    assert pc.all(pc.equal(result_2["a"], expected["a"])).as_py()

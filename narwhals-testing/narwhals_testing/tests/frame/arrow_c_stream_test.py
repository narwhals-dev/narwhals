from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, PYARROW_VERSION

pytest.importorskip("polars")
pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.compute")

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc


@pytest.mark.skipif(POLARS_VERSION < (1, 3), reason="too old for pycapsule in Polars")
@pytest.mark.skipif(
    PYARROW_VERSION < (16, 0, 0), reason="too old for pycapsule in PyArrow"
)
def test_arrow_c_stream_test() -> None:
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    result = pa.table(df)
    expected = pa.table({"a": [1, 2, 3]})
    assert pc.all(pc.equal(result["a"], expected["a"])).as_py()


@pytest.mark.skipif(POLARS_VERSION < (1, 3), reason="too old for pycapsule in Polars")
@pytest.mark.skipif(
    PYARROW_VERSION < (16, 0, 0), reason="too old for pycapsule in PyArrow"
)
def test_arrow_c_stream_test_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    # "poison" the dunder method to make sure it actually got called above
    monkeypatch.setattr(
        "narwhals.dataframe.DataFrame.__arrow_c_stream__", lambda *_: 1 / 0
    )
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        pa.table(df)


@pytest.mark.skipif(POLARS_VERSION < (1, 3), reason="too old for pycapsule in Polars")
@pytest.mark.skipif(
    PYARROW_VERSION < (16, 0, 0), reason="too old for pycapsule in PyArrow"
)
def test_arrow_c_stream_test_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Check that fallback to PyArrow works
    monkeypatch.delattr("polars.DataFrame.__arrow_c_stream__")
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    result = pa.table(df)
    expected = pa.table({"a": [1, 2, 3]})
    assert pc.all(pc.equal(result["a"], expected["a"])).as_py()
    result_2 = nw.from_arrow(result, backend="pandas").to_arrow()
    assert pc.all(pc.equal(result_2["a"], expected["a"])).as_py()

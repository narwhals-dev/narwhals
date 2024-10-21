from __future__ import annotations

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import narwhals.stable.v1 as nw


def test_arrow_c_stream_test(
    request: pytest.FixtureRequest,
    polars_version: tuple[int, ...],
    pyarrow_version: tuple[int, ...],
) -> None:
    if polars_version < (1, 3):
        request.applymarker(pytest.mark.skipif(reason="too old for pycapsule in Polars"))
    if pyarrow_version < (16, 0, 0):
        request.applymarker(pytest.mark.skipif(reason="too old for pycapsule in PyArrow"))
    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result = pa.chunked_array(s)
    expected = pa.chunked_array([[1, 2, 3]])
    assert pc.all(pc.equal(result, expected)).as_py()


def test_arrow_c_stream_test_invalid(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    polars_version: tuple[int, ...],
    pyarrow_version: tuple[int, ...],
) -> None:
    if polars_version < (1, 3):
        request.applymarker(pytest.mark.skipif(reason="too old for pycapsule in Polars"))
    if pyarrow_version < (16, 0, 0):
        request.applymarker(pytest.mark.skipif(reason="too old for pycapsule in PyArrow"))
    # "poison" the dunder method to make sure it actually got called above
    monkeypatch.setattr("narwhals.series.Series.__arrow_c_stream__", lambda *_: 1 / 0)
    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        pa.chunked_array(s)


def test_arrow_c_stream_test_fallback(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    polars_version: tuple[int, ...],
    pyarrow_version: tuple[int, ...],
) -> None:
    if polars_version < (1, 3):
        request.applymarker(pytest.mark.skipif(reason="too old for pycapsule in Polars"))
    if pyarrow_version < (16, 0, 0):
        request.applymarker(pytest.mark.skipif(reason="too old for pycapsule in PyArrow"))
    # Check that fallback to PyArrow works
    monkeypatch.delattr("polars.Series.__arrow_c_stream__")
    s = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)["a"]
    s.__arrow_c_stream__()
    result = pa.chunked_array(s)
    expected = pa.chunked_array([[1, 2, 3]])
    assert pc.all(pc.equal(result, expected)).as_py()

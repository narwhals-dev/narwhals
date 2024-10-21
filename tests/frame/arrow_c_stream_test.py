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
    if polars_version < (1, 3):  # pragma: no cover
        request.applymarker(pytest.mark.skip(reason="too old for pycapsule in Polars"))
    if pyarrow_version < (16, 0, 0):  # pragma: no cover
        request.applymarker(pytest.mark.skip(reason="too old for pycapsule in PyArrow"))
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    result = pa.table(df)
    expected = pa.table({"a": [1, 2, 3]})
    assert pc.all(pc.equal(result["a"], expected["a"])).as_py()


def test_arrow_c_stream_test_invalid(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    polars_version: tuple[int, ...],
    pyarrow_version: tuple[int, ...],
) -> None:
    if polars_version < (1, 3):  # pragma: no cover
        request.applymarker(pytest.mark.skip(reason="too old for pycapsule in Polars"))
    if pyarrow_version < (16, 0, 0):  # pragma: no cover
        request.applymarker(pytest.mark.skip(reason="too old for pycapsule in PyArrow"))
    # "poison" the dunder method to make sure it actually got called above
    monkeypatch.setattr(
        "narwhals.dataframe.DataFrame.__arrow_c_stream__", lambda *_: 1 / 0
    )
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        pa.table(df)


def test_arrow_c_stream_test_fallback(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    polars_version: tuple[int, ...],
    pyarrow_version: tuple[int, ...],
) -> None:
    if polars_version < (1, 3):  # pragma: no cover
        request.applymarker(pytest.mark.skip(reason="too old for pycapsule in Polars"))
    if pyarrow_version < (16, 0, 0):  # pragma: no cover
        request.applymarker(pytest.mark.skip(reason="too old for pycapsule in PyArrow"))
    # Check that fallback to PyArrow works
    monkeypatch.delattr("polars.DataFrame.__arrow_c_stream__")
    df = nw.from_native(pl.Series([1, 2, 3]).to_frame("a"), eager_only=True)
    result = pa.table(df)
    expected = pa.table({"a": [1, 2, 3]})
    assert pc.all(pc.equal(result["a"], expected["a"])).as_py()

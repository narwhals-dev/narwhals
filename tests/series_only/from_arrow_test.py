from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from packaging.version import Version as PkgVersion

pytest.importorskip("pyarrow")
import pyarrow as pa

import narwhals as nw
from tests.utils import PYARROW_VERSION, assert_equal_series

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed

NAME = "a"
DATA = [5, None, 0, 1]


class ArrayOnly:
    """Exposes only `__arrow_c_array__`, hiding the native `pyarrow.Array`."""

    def __init__(self, arr: pa.Array[Any]) -> None:
        self._arr = arr

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]:
        capsules: tuple[object, object] = self._arr.__arrow_c_array__(requested_schema)
        return capsules


@pytest.mark.parametrize(
    "values", [pa.array(DATA), pa.chunked_array([DATA[:2], DATA[2:]])], ids=type
)
def test_series_from_arrow_native(
    eager_backend: EagerAllowed, values: pa.Array[Any] | pa.ChunkedArray[Any]
) -> None:
    result = nw.Series.from_arrow(NAME, values, backend=eager_backend)
    assert isinstance(result, nw.Series)
    assert_equal_series(result, DATA, NAME)


@pytest.mark.parametrize(
    "values", [pa.array(DATA), pa.chunked_array([DATA[:2], DATA[2:]])], ids=type
)
def test_series_from_arrow_native_without_pycapsule(
    eager_backend: EagerAllowed,
    values: pa.Array[Any] | pa.ChunkedArray[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate `pyarrow<14`, where native arrays don't have the PyCapsule methods
    monkeypatch.setattr("narwhals.series.supports_arrow_c_array", lambda _: False)
    monkeypatch.setattr("narwhals.series.supports_arrow_c_stream", lambda _: False)
    result = nw.Series.from_arrow(NAME, values, backend=eager_backend)
    assert_equal_series(result, DATA, NAME)


@pytest.mark.skipif(PYARROW_VERSION < PkgVersion("14"), reason="too old")
def test_series_from_arrow_c_array(eager_backend: EagerAllowed) -> None:
    values = ArrayOnly(pa.array(DATA))
    result = nw.Series.from_arrow(NAME, values, backend=eager_backend)
    assert_equal_series(result, DATA, NAME)


@pytest.mark.skipif(PYARROW_VERSION < PkgVersion("16"), reason="too old")
def test_series_from_arrow_c_stream(eager_backend: EagerAllowed) -> None:
    values = nw.from_native(pa.chunked_array([DATA]), series_only=True)
    result = nw.Series.from_arrow(NAME, values, backend=eager_backend)
    assert_equal_series(result, DATA, NAME)


@pytest.mark.skipif(PYARROW_VERSION < PkgVersion("16"), reason="too old")
def test_series_from_arrow_c_stream_keeps_chunks() -> None:
    # `nw.Series` exposes both PyCapsule methods; the stream is preferred so that
    # chunks are not combined.
    native = pa.chunked_array([DATA[:2], DATA[2:]])
    values = nw.from_native(native, series_only=True)
    result = nw.Series.from_arrow(NAME, values, backend="pyarrow")
    assert result.to_native().num_chunks == native.num_chunks
    assert_equal_series(result, DATA, NAME)


@pytest.mark.skipif(PYARROW_VERSION < PkgVersion("16"), reason="too old")
def test_series_from_arrow_table_is_struct(eager_backend: EagerAllowed) -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    result = nw.Series.from_arrow(NAME, table, backend=eager_backend)
    assert result.len() == 2
    assert result.to_list() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


@pytest.mark.skipif(PYARROW_VERSION < PkgVersion("16"), reason="too old")
def test_series_from_arrow_renames_polars() -> None:
    pl = pytest.importorskip("polars")
    result = nw.Series.from_arrow(NAME, pl.Series("b", DATA), backend="pyarrow")
    assert result.implementation.is_pyarrow()
    assert_equal_series(result, DATA, NAME)


def test_series_from_arrow_invalid() -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        nw.Series.from_arrow(NAME, DATA, backend="pyarrow")  # type: ignore[arg-type]
    pytest.importorskip("sqlframe")
    with pytest.raises(ValueError, match="lazy"):
        nw.Series.from_arrow(NAME, pa.array(DATA), backend="sqlframe")  # type: ignore[arg-type]

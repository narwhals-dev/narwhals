from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

# A null map has no length, but a null value still counts as an entry.
expected = {"a": [2, 0, None, 1]}


@pytest.fixture
def map_series() -> pl.Series:
    pytest.importorskip("polars", minversion="2.0")
    import polars as pl

    return pl.Series(
        "a", [{"x": 1, "y": 2}, {}, None, {"x": None}], dtype=pl.Map(pl.String, pl.Int64)
    )


def test_len_series(map_series: pl.Series) -> None:
    series = nw.from_native(map_series, series_only=True)
    result = series.map.len()

    assert_equal_data({"a": result}, expected)
    assert result.dtype == nw.UInt32


def test_len_expr_eager(map_series: pl.Series) -> None:
    frame = nw.from_native(map_series.to_frame())
    result = frame.select(nw.col("a").map.len())

    assert_equal_data(result, expected)
    assert result.schema["a"] == nw.UInt32


def test_len_expr_lazy(map_series: pl.Series) -> None:
    frame = nw.from_native(map_series.to_frame().lazy())
    result = frame.select(nw.col("a").map.len())

    assert_equal_data(result, expected)
    assert result.collect_schema()["a"] == nw.UInt32


@pytest.fixture
def arrow_map_table() -> pa.Table:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    maps = pa.array(
        [[("x", 1), ("y", 2)], [], None, [("x", None)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    return pa.table({"a": maps})


def test_len_arrow_series(arrow_map_table: pa.Table) -> None:
    series = nw.from_native(arrow_map_table).get_column("a")
    result = series.map.len()

    assert_equal_data({"a": result}, expected)
    assert result.name == "a"
    assert result.dtype == nw.UInt32


def test_len_arrow_expr(arrow_map_table: pa.Table) -> None:
    frame = nw.from_native(arrow_map_table)
    result = frame.select(nw.col("a").map.len())

    assert_equal_data(result, expected)
    assert result.schema == {"a": nw.UInt32}


def test_len_arrow_sliced_chunks() -> None:
    pa = pytest.importorskip("pyarrow")
    maps = pa.array(
        [[("pad", 0)], [("x", 1), ("x", 2)], [("y", None)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    validity = pa.array([0, None, 0], type=pa.int32()).buffers()[0]
    # The null map still spans two entries in the offsets buffer.
    masked = pa.Array.from_buffers(
        maps.type,
        len(maps),
        [validity, maps.buffers()[1]],
        null_count=1,
        children=[maps.values],
    )
    chunks = pa.chunked_array([maps.slice(1), masked.slice(1), maps.slice(2, 0)])
    series = nw.from_native(chunks, series_only=True)
    result = series.map.len().to_native()

    assert result.to_pylist() == [2, 1, None, 1]
    assert result.type == pa.uint32()
    assert [len(chunk) for chunk in result.chunks] == [2, 2, 0]


def test_len_arrow_zero_chunks() -> None:
    pa = pytest.importorskip("pyarrow")
    chunks = pa.chunked_array([], type=pa.map_(pa.string(), pa.int64()))
    series = nw.from_native(chunks, series_only=True)
    result = series.map.len().to_native()

    assert len(result) == 0
    assert result.num_chunks == 0
    assert result.type == pa.uint32()


@pytest.mark.skipif(
    POLARS_VERSION >= PkgVersion("2"), reason="native Map dtype requires Polars 2.0"
)
def test_len_polars_unsupported() -> None:  # pragma: no cover - Polars 2 in coverage CI.
    pl = pytest.importorskip("polars")
    frame = nw.from_native(pl.DataFrame({"a": [1]}))
    msg = "`Map` dtype is only supported for 'polars>=2.0'"

    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        frame.select(nw.col("a").map.len())

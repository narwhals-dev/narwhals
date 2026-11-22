from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.expr_and_series.map.get_test import map_frame
from tests.utils import POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from tests.utils import ConstructorPandasLike

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


def test_len_duckdb_expr(arrow_map_table: pa.Table) -> None:
    pytest.importorskip("duckdb")
    import duckdb
    import pyarrow as pa

    row_ids = [0, 1, 2, 3]
    table = arrow_map_table.append_column("row_id", pa.array(row_ids))
    frame = nw.from_native(duckdb.from_arrow(table))
    result = frame.select("row_id", nw.col("a").map.len()).sort("row_id")

    assert_equal_data(result, {"row_id": row_ids, **expected})
    assert result.collect_schema()["a"] == nw.UInt32


@pytest.mark.parametrize("list_size", [-1, 2], ids=["list", "array"])
def test_len_duckdb_non_map(list_size: int) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    import duckdb
    import pyarrow as pa

    data = pa.array([[1, 2]], type=pa.list_(pa.int64(), list_size))
    frame = nw.from_native(duckdb.from_arrow(pa.table({"a": data})))

    with pytest.raises(
        (duckdb.InvalidInputException, duckdb.BinderException),
        match=r"(?i)not a map|map_keys\(",
    ):
        frame.select(nw.col("a").map.len())


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
        children=[maps.values],  # noqa: PD011 - Arrow map child array.
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


@pytest.fixture
def pandas_like_map_frame(
    request: pytest.FixtureRequest, constructor_pandas_like: ConstructorPandasLike
) -> pd.DataFrame:
    pytest.importorskip("pandas", minversion="2.2")
    import pandas as pd

    data = [[("x", 1), ("x", 2)], [], None, [("x", None)]]
    frame: pd.DataFrame = map_frame(request, constructor_pandas_like, data)
    frame.index = pd.Index([9, 4, 9, 2], name="row")
    return frame


def test_len_pandas_like_series(pandas_like_map_frame: pd.DataFrame) -> None:
    import pandas as pd
    import pyarrow as pa

    series = nw.from_native(pandas_like_map_frame["a"], series_only=True)
    result = series.map.len()
    native = result.to_native()

    assert_equal_data({"a": result}, expected)
    assert result.dtype == nw.UInt32
    assert native.dtype == pd.ArrowDtype(pa.uint32())
    assert native.name == series.name
    pd.testing.assert_index_equal(
        native.index, pandas_like_map_frame.index, check_names=True
    )


def test_len_pandas_like_expr(pandas_like_map_frame: pd.DataFrame) -> None:
    import pandas as pd
    import pyarrow as pa

    frame = nw.from_native(pandas_like_map_frame)
    result = frame.select(nw.col("a").map.len())
    native = result.to_native()

    assert_equal_data(result, expected)
    assert result.schema == {"a": nw.UInt32}
    assert native["a"].dtype == pd.ArrowDtype(pa.uint32())
    assert native["a"].name == "a"
    pd.testing.assert_index_equal(
        native.index, pandas_like_map_frame.index, check_names=True
    )


def test_len_pandas_non_pyarrow() -> None:
    pd = pytest.importorskip("pandas")
    series = nw.from_native(pd.Series([{"x": 1}], dtype=object), series_only=True)
    msg = re.escape("Series must be of PyArrow Map type to support map namespace.")

    with pytest.raises(TypeError, match=msg):
        series.map.len()


def test_len_pandas_unsupported_version(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas", minversion="2.0")
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setattr("narwhals._pandas_like.series_map.PANDAS_VERSION", (2, 1, 4))

    maps = pa.array([[("x", 1)]], type=pa.map_(pa.string(), pa.int64()))
    native = pd.Series(maps, dtype=pd.ArrowDtype(maps.type), name="a")
    series = nw.from_native(native, series_only=True)
    frame = nw.from_native(native.to_frame())
    msg = re.escape("`map.len()` requires pandas 2.2 or newer.")

    with pytest.raises(NotImplementedError, match=msg):
        series.map.len()
    with pytest.raises(NotImplementedError, match=msg):
        frame.select(nw.col("a").map.len())


@pytest.mark.skipif(
    POLARS_VERSION >= PkgVersion("2"), reason="native Map dtype requires Polars 2.0"
)
def test_len_polars_unsupported() -> None:  # pragma: no cover - Polars 2 in coverage CI.
    pl = pytest.importorskip("polars")
    frame = nw.from_native(pl.DataFrame({"a": [1]}))
    msg = "`Map` dtype is only supported for 'polars>=2.0'"

    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        frame.select(nw.col("a").map.len())

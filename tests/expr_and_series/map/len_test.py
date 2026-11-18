from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    import polars as pl

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


def test_len_arrow_unsupported() -> None:
    pa = pytest.importorskip("pyarrow")
    values = pa.array([[("x", 1)]], type=pa.map_(pa.string(), pa.int64()))
    frame = nw.from_native(pa.table({"a": values}))

    with pytest.raises(NotImplementedError, match="'len' is not implemented"):
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

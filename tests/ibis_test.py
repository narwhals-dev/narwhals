from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    import ibis

    from tests.utils import Constructor
else:
    ibis = pytest.importorskip("ibis")


@pytest.fixture
def ibis_constructor() -> Constructor:
    pl = pytest.importorskip("polars")

    def func(data: dict[str, Any]) -> ibis.Table:
        df = pl.DataFrame(data)
        return ibis.memtable(df)

    return func


def test_from_native(ibis_constructor: Constructor) -> None:
    df = nw.from_native(ibis_constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.columns == ["a", "b"]


def test_map_len() -> None:
    pa = pytest.importorskip("pyarrow")
    maps = pa.array(
        [[("x", 1), ("y", 2)], [], None, [("x", None)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    row_ids = [0, 1, 2, 3]
    frame = nw.from_native(pa.table({"row_id": row_ids, "a": maps})).lazy("ibis")
    result = frame.select("row_id", nw.col("a").map.len()).sort("row_id")

    assert result.collect_schema()["a"] == nw.UInt32
    collected = result.collect("pyarrow").to_native()
    assert collected.to_pydict() == {"row_id": row_ids, "a": [2, 0, None, 1]}
    assert collected["a"].type == pa.uint32()


def test_map_len_non_map() -> None:
    table = ibis.table({"a": "array<int64>"}, name="arrays")
    frame: nw.LazyFrame[ibis.Table] = nw.from_native(table)
    with pytest.raises(TypeError, match=r"`map\.len\(\)` requires a Map expression\."):
        frame.select(nw.col("a").map.len())

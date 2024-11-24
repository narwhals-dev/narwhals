from __future__ import annotations

import duckdb
import polars as pl
import pytest

import narwhals.stable.v1 as nw


def test_write_csv_duckdb(tmpdir: pytest.TempdirFactory) -> None:
    data = {"a": [1, 2, 3]}
    path = tmpdir / "foo.csv"  # type: ignore[operator]
    _df = pl.DataFrame(data)
    result = nw.from_native(duckdb.table("_df")).write_csv(str(path))  # type: ignore[union-attr]
    content = path.read_text("utf-8")
    assert path.exists()
    assert result is None
    assert content == "a\n1\n2\n3\n"
    with pytest.raises(NotImplementedError, match="write_csv"):
        nw.from_native(duckdb.table("_df")).write_csv()  # type: ignore[union-attr]


def test_write_csv_ibis(tmpdir: pytest.TempdirFactory) -> None:
    ibis = pytest.importorskip("ibis")
    data = {"a": [1, 2, 3]}
    path = tmpdir / "foo.csv"  # type: ignore[operator]
    df = pl.DataFrame(data)
    result = nw.from_native(ibis.memtable(df)).write_csv(str(path))  # type: ignore[union-attr]
    content = path.read_text("utf-8")
    assert path.exists()
    assert result is None
    assert content == "a\n1\n2\n3\n"
    with pytest.raises(NotImplementedError, match="write_csv"):
        nw.from_native(ibis.memtable(df)).write_csv()  # type: ignore[union-attr]

from __future__ import annotations

import polars as pl
import pytest

import narwhals.stable.v1 as nw

data = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}


def test_interchange() -> None:
    df_pl = pl.DataFrame(data)
    df = nw.from_native(df_pl.__dataframe__(), eager_or_interchange_only=True)
    series = df["a"]

    with pytest.raises(
        NotImplementedError,
        match="Cannot access native namespace for metadata-only dataframes with unknown backend",
    ):
        df.__native_namespace__()

    with pytest.raises(
        NotImplementedError,
        match="Cannot access native namespace for metadata-only series with unknown backend",
    ):
        series.__native_namespace__()


@pytest.mark.filterwarnings("ignore:.*The `ArrowDtype` class is not available in pandas")
def test_ibis(
    tmpdir: pytest.TempdirFactory,
) -> None:  # pragma: no cover
    ibis = pytest.importorskip("ibis")
    df_pl = pl.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    tbl = ibis.read_parquet(filepath)
    df = nw.from_native(tbl, eager_or_interchange_only=True)
    series = df["a"]

    assert df.__native_namespace__() == ibis
    assert series.__native_namespace__() == ibis


def test_duckdb() -> None:
    duckdb = pytest.importorskip("duckdb")
    df_pl = pl.DataFrame(data)  # noqa: F841

    rel = duckdb.sql("select * from df_pl")
    df = nw.from_native(rel, eager_or_interchange_only=True)
    series = df["a"]

    assert df.__native_namespace__() == duckdb
    assert series.__native_namespace__() == duckdb

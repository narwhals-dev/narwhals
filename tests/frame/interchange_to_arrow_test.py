import duckdb
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw

data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.1], "z": ["x", "y", "z"]}


def test_interchange_to_arrow() -> None:
    df_pl = pl.DataFrame(data)
    df = nw.from_native(df_pl.__dataframe__(), eager_or_interchange_only=True)
    result = df.to_arrow()

    assert isinstance(result, pa.Table)


def test_interchange_ibis_to_arrow(
    tmpdir: pytest.TempdirFactory,
) -> None:  # pragma: no cover
    ibis = pytest.importorskip("ibis")
    df_pl = pl.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)

    tbl = ibis.read_parquet(filepath)
    df = nw.from_native(tbl, eager_or_interchange_only=True)
    result = df.to_arrow()

    assert isinstance(result, pa.Table)


def test_interchange_duckdb_to_arrow() -> None:
    df_pl = pl.DataFrame(data)  # noqa: F841
    rel = duckdb.sql("select * from df_pl")
    df = nw.from_native(rel, eager_or_interchange_only=True)
    result = df.to_arrow()

    assert isinstance(result, pa.Table)

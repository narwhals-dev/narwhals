import duckdb
import pandas as pd
import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "z": ["x", "y", "z"]}


def test_interchange_to_pandas(request: pytest.FixtureRequest) -> None:
    if parse_version(pd.__version__) < parse_version("2.0.2"):
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)
    df = nw.from_native(df_pl.__dataframe__(), eager_or_interchange_only=True)

    assert df.to_pandas().equals(df_pl.to_pandas())


def test_interchange_ibis_to_pandas(
    tmpdir: pytest.TempdirFactory, request: pytest.FixtureRequest
) -> None:  # pragma: no cover
    if parse_version(pd.__version__) < parse_version("2.0.2"):
        request.applymarker(pytest.mark.xfail)

    ibis = pytest.importorskip("ibis")
    df_pl = pl.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)

    tbl = ibis.read_parquet(filepath)
    df = nw.from_native(tbl, eager_or_interchange_only=True)

    assert df.to_pandas().equals(df_pl.to_pandas())


def test_interchange_duckdb_to_pandas() -> None:
    df_pl = pl.DataFrame(data)
    rel = duckdb.sql("select * from df_pl")
    df = nw.from_native(rel, eager_or_interchange_only=True)

    assert df.to_pandas().equals(df_pl.to_pandas())

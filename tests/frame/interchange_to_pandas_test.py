from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import narwhals.stable.v1 as nw

data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "z": ["x", "y", "z"]}


def test_interchange_to_pandas(
    request: pytest.FixtureRequest, pandas_version: tuple[int, ...]
) -> None:
    if pandas_version < (1, 5, 0):
        request.applymarker(pytest.mark.xfail)
    df_raw = pd.DataFrame(data)
    df = nw.from_native(df_raw.__dataframe__(), eager_or_interchange_only=True)

    assert df.to_pandas().equals(df_raw)


def test_interchange_ibis_to_pandas(
    tmpdir: pytest.TempdirFactory,
    request: pytest.FixtureRequest,
    pandas_version: tuple[int, ...],
) -> None:  # pragma: no cover
    if pandas_version < (1, 5, 0):
        request.applymarker(pytest.mark.xfail)

    ibis = pytest.importorskip("ibis")
    df_raw = pd.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_raw.to_parquet(filepath)

    tbl = ibis.read_parquet(filepath)
    df = nw.from_native(tbl, eager_or_interchange_only=True)

    assert df.to_pandas().equals(df_raw)


def test_interchange_duckdb_to_pandas(
    request: pytest.FixtureRequest, pandas_version: tuple[int, ...]
) -> None:
    if pandas_version < (1, 0, 0):
        request.applymarker(pytest.mark.xfail)
    df_raw = pd.DataFrame(data)
    rel = duckdb.sql("select * from df_raw")
    df = nw.from_native(rel, eager_or_interchange_only=True)

    assert df.to_pandas().equals(df_raw)

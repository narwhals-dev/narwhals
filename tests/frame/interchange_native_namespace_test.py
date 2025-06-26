from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals.stable.v1 as nw_v1

pytest.importorskip("polars")
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["x", "y", "w"]}


def test_interchange() -> None:
    df_pl = pl.DataFrame(data)
    df = nw_v1.from_native(df_pl.__dataframe__(), eager_or_interchange_only=True)
    series = df["a"]

    with pytest.raises(
        NotImplementedError,
        match="Cannot access native namespace for interchange-level dataframes with unknown backend",
    ):
        df.__native_namespace__()

    with pytest.raises(
        NotImplementedError,
        match="Cannot access native namespace for interchange-level series with unknown backend",
    ):
        series.__native_namespace__()


@pytest.mark.filterwarnings("ignore:.*The `ArrowDtype` class is not available in pandas")
def test_ibis(
    tmpdir: pytest.TempdirFactory, request: pytest.FixtureRequest
) -> None:  # pragma: no cover
    pytest.importorskip("ibis")
    import ibis

    try:
        ibis.set_backend("duckdb")
    except ImportError:
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    tbl = ibis.read_parquet(filepath)
    df = nw_v1.from_native(tbl, eager_or_interchange_only=True)
    series = df["a"]

    assert df.__native_namespace__() == ibis
    assert series.__native_namespace__() == ibis


def test_duckdb() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    df_pl = pl.DataFrame(data)  # noqa: F841

    rel = duckdb.sql("select * from df_pl")
    df = nw_v1.from_native(rel, eager_or_interchange_only=True)
    series = df["a"]

    assert df.__native_namespace__() == duckdb
    assert series.__native_namespace__() == duckdb

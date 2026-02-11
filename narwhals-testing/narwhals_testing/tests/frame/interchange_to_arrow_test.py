from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals.stable.v1 as nw_v1

if TYPE_CHECKING:
    from collections.abc import Mapping

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.1], "z": ["x", "y", "z"]}

pytest.importorskip("polars")
pytest.importorskip("pyarrow")


def test_interchange_to_arrow() -> None:
    import polars as pl
    import pyarrow as pa

    df_pl = pl.DataFrame(data)
    df = nw_v1.from_native(df_pl.__dataframe__(), eager_or_interchange_only=True)
    result = df.to_arrow()

    assert isinstance(result, pa.Table)


def test_interchange_ibis_to_arrow(
    tmpdir: pytest.TempdirFactory, request: pytest.FixtureRequest
) -> None:  # pragma: no cover
    pytest.importorskip("ibis")

    import ibis
    import polars as pl
    import pyarrow as pa

    try:
        ibis.set_backend("duckdb")
    except ImportError:
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)

    tbl = ibis.read_parquet(filepath)
    df = nw_v1.from_native(tbl, eager_or_interchange_only=True)
    result = df.to_arrow()

    assert isinstance(result, pa.Table)


def test_interchange_duckdb_to_arrow() -> None:
    pytest.importorskip("duckdb")

    import duckdb
    import polars as pl
    import pyarrow as pa

    df_pl = pl.DataFrame(data)  # noqa: F841
    rel = duckdb.sql("select * from df_pl")
    df = nw_v1.from_native(rel, eager_or_interchange_only=True)
    result = df.to_arrow()

    assert isinstance(result, pa.Table)

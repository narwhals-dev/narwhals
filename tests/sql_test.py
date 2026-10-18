from __future__ import annotations

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import DUCKDB_VERSION


def test_sql() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("sqlparse")
    if DUCKDB_VERSION < PkgVersion("1.3"):
        pytest.skip()
    from narwhals.sql import table

    schema = {"date": nw.Date(), "price": nw.Int64(), "symbol": nw.String()}
    assets = table("assets", schema)
    result = assets.with_columns(
        returns=(nw.col("price") / nw.col("price").shift(1)).over(
            "symbol", order_by="date"
        )
    )
    # DuckDB 2.0 qualifies table names with the catalog.
    table_name = (
        "memory.main.assets" if DUCKDB_VERSION >= PkgVersion("2") else "main.assets"
    )
    expected = f"""SELECT date, price, symbol, (price / lag(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC NULLS FIRST)) AS "returns" FROM {table_name}"""  # noqa: S608
    assert result.to_sql() == expected
    expected = (
        "SELECT date, price,\n"
        "             symbol,\n"
        "             (price / lag(price, 1) OVER (PARTITION BY symbol\n"
        '                                          ORDER BY date ASC NULLS FIRST)) AS "returns"\n'
        f"FROM {table_name}"
    )
    assert result.to_sql(pretty=True) == expected


def test_sql_table_schema_pairs() -> None:
    pytest.importorskip("duckdb")
    if DUCKDB_VERSION < PkgVersion("1.3"):
        pytest.skip()
    from narwhals.sql import table

    result = table("assets_pairs", [("date", nw.Date), ("price", nw.Int64())])
    assert result.collect_schema() == {"date": nw.Date(), "price": nw.Int64()}

from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION


def test_sql() -> None:
    pytest.importorskip("duckdb")
    if DUCKDB_VERSION < (1, 3):
        pytest.skip()
    from narwhals.sql import table

    schema = {"date": nw.Date, "price": nw.Int64, "symbol": nw.String}
    assets = table("assets", schema)
    result = (
        assets.with_columns(
            returns=(nw.col("price") / nw.col("price").shift(1)).over(
                "symbol", order_by="date"
            )
        )
        .to_native()
        .sql_query()
    )
    expected = """SELECT date, price, symbol, (price / lag(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC NULLS FIRST)) AS "returns" FROM main.assets"""
    assert result == expected

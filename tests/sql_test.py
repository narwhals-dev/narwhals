from __future__ import annotations

import uuid

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION


def _unique_name(prefix: str) -> str:
    # `narwhals.sql.table` creates the table in a process-wide catalog, so
    # names must be unique for tests to be re-runnable and thread-safe.
    return f"{prefix}_{uuid.uuid4().hex}"


def test_sql() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("sqlparse")
    if DUCKDB_VERSION < (1, 3):
        pytest.skip()
    from narwhals.sql import table

    name = _unique_name("assets")
    schema = {"date": nw.Date(), "price": nw.Int64(), "symbol": nw.String()}
    assets = table(name, schema)
    result = assets.with_columns(
        returns=(nw.col("price") / nw.col("price").shift(1)).over(
            "symbol", order_by="date"
        )
    )
    expected = f"""SELECT date, price, symbol, (price / lag(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC NULLS FIRST)) AS "returns" FROM main.{name}"""  # noqa: S608
    assert result.to_sql() == expected
    expected = (
        "SELECT date, price,\n"
        "             symbol,\n"
        "             (price / lag(price, 1) OVER (PARTITION BY symbol\n"
        '                                          ORDER BY date ASC NULLS FIRST)) AS "returns"\n'
        f"FROM main.{name}"
    )
    assert result.to_sql(pretty=True) == expected


def test_sql_table_schema_pairs() -> None:
    pytest.importorskip("duckdb")
    if DUCKDB_VERSION < (1, 3):
        pytest.skip()
    from narwhals.sql import table

    name = _unique_name("assets_pairs")
    result = table(name, [("date", nw.Date), ("price", nw.Int64())])
    assert result.collect_schema() == {"date": nw.Date(), "price": nw.Int64()}

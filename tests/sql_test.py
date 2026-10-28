from __future__ import annotations

import uuid

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import DUCKDB_VERSION

pytest.importorskip("duckdb", minversion="1.3.0")

from narwhals.sql import table


def _unique_name(prefix: str) -> str:
    # `narwhals.sql.table` creates the table in a process-wide catalog, so
    # names must be unique for tests to be re-runnable and thread-safe.
    return f"{prefix}_{uuid.uuid4().hex}"


def test_sql() -> None:
    pytest.importorskip("sqlparse")

    name = _unique_name("assets")
    schema = {"date": nw.Date(), "price": nw.Int64(), "symbol": nw.String()}
    assets = table(name, schema)
    result = assets.with_columns(
        returns=(nw.col("price") / nw.col("price").shift(1)).over(
            "symbol", order_by="date"
        )
    )
    # DuckDB 2.0 renders the relation's own SQL instead of the parsed query.
    if DUCKDB_VERSION >= PkgVersion("2"):  # pragma: no cover
        expected = (
            'SELECT "date", "price", "symbol", ("price" / (\n'  # noqa: S608
            'lag("price", 1) over (partition by "symbol" order by "date" asc nulls first )\n'
            f')) AS "returns" FROM "{name}"'
        )
        expected_pretty = (
            'SELECT "date",\n'
            '       "price",\n'
            '       "symbol",\n'
            '       ("price" / (lag("price", 1) OVER (PARTITION BY "symbol"\n'
            '                                         ORDER BY "date" ASC NULLS FIRST))) AS "returns"\n'
            f'FROM "{name}"'
        )
    else:  # pragma: no cover
        expected = f"""SELECT date, price, symbol, (price / lag(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC NULLS FIRST)) AS "returns" FROM main.{name}"""  # noqa: S608
        expected_pretty = (
            "SELECT date, price,\n"
            "             symbol,\n"
            "             (price / lag(price, 1) OVER (PARTITION BY symbol\n"
            '                                          ORDER BY date ASC NULLS FIRST)) AS "returns"\n'
            f"FROM main.{name}"
        )
    assert result.to_sql() == expected
    assert result.to_sql(pretty=True) == expected_pretty


def test_sql_table_schema_pairs() -> None:
    name = _unique_name("assets_pairs")
    result = table(name, [("date", nw.Date), ("price", nw.Int64())])
    assert result.collect_schema() == {"date": nw.Date(), "price": nw.Int64()}


def test_sql_table_combine() -> None:
    """Tables created in the same thread must be combinable."""
    lhs_name, rhs_name = _unique_name("lhs"), _unique_name("rhs")
    lhs = table(lhs_name, {"id": nw.Int64(), "x": nw.Int64()})
    rhs = table(rhs_name, {"id": nw.Int64(), "y": nw.Int64()})

    joined = lhs.join(rhs, on="id", how="inner").to_sql()
    assert lhs_name in joined
    assert rhs_name in joined

    # `nw.concat` is typed as a plain `LazyFrame`, so go through the relation.
    concatenated = nw.concat([lhs.select("id"), rhs.select("id")]).to_native().sql_query()
    assert lhs_name in concatenated
    assert rhs_name in concatenated

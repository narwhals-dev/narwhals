from __future__ import annotations

import pytest

import narwhals as nw


def test_sql() -> None:
    pytest.importorskip("sqlframe")
    schema = {"date": nw.Date, "price": nw.Int64, "symbol": nw.String}
    assets = nw.sql.table("assets", schema)
    result = (
        assets.with_columns(
            nw.col("price").rolling_mean(5).over("symbol", order_by="date")
        )
        .to_native()
        .sql("duckdb")
    )
    expected = (
        "SELECT\n"
        '  "assets"."date" AS "date",\n'
        "  CASE\n"
        '    WHEN COUNT("assets"."price") OVER (\n'
        '      PARTITION BY "assets"."symbol"\n'
        '      ORDER BY "assets"."date" NULLS FIRST\n'
        "      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW\n"
        "    ) >= 5\n"
        '    THEN AVG("assets"."price") OVER (\n'
        '      PARTITION BY "assets"."symbol"\n'
        '      ORDER BY "assets"."date" NULLS FIRST\n'
        "      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW\n"
        "    )\n"
        '  END AS "price",\n'
        '  "assets"."symbol" AS "symbol"\n'
        'FROM "assets" AS "assets"'
    )
    assert result == expected

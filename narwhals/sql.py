from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals.translate import from_native
from narwhals.utils import Version

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation

    from narwhals.dataframe import LazyFrame
    from narwhals.typing import IntoSchema

try:
    import duckdb  # ignore-banned-import
except ImportError as exc:  # pragma: no cover
    msg = (
        "`narwhals.sql` requires DuckDB to be installed.\n\n"
        "Hint: run `pip install -U narwhals[sql]`"
    )
    raise ModuleNotFoundError(msg) from exc

conn = duckdb.connect()
tz = conn.sql("select value from duckdb_settings() where name = 'TimeZone'").fetchone()[0]


def table(name: str, schema: IntoSchema) -> LazyFrame[DuckDBPyRelation]:
    """Generate standalone LazyFrame which you can use to generate SQL.

    Note that this requires DuckDB to be installed.

    Parameters:
        name: Table name.
        schema: Table schema.

    Returns:
        A LazyFrame.

    Examples:
        >>> import narwhals as nw
        >>> schema = {"date": nw.Date, "price": nw.Int64, "symbol": nw.String}
        >>> assets = nw.sql.table("assets", schema)
        >>> result = assets.with_columns(
        ...     nw.col("price").rolling_mean(5).over("symbol", order_by="date")
        ... )
        >>> print(result.to_native().sql(dialect="duckdb"))
        SELECT
          "assets"."date" AS "date",
          CASE
            WHEN COUNT("assets"."price") OVER (
              PARTITION BY "assets"."symbol"
              ORDER BY "assets"."date" NULLS FIRST
              ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) >= 5
            THEN AVG("assets"."price") OVER (
              PARTITION BY "assets"."symbol"
              ORDER BY "assets"."date" NULLS FIRST
              ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )
          END AS "price",
          "assets"."symbol" AS "symbol"
        FROM "assets" AS "assets"
    """
    column_mapping = {
        col: narwhals_to_native_dtype(dtype, Version.MAIN, tz)
        for col, dtype in schema.items()
    }
    dtypes = ", ".join(f"{col} {dtype}" for col, dtype in column_mapping.items())
    conn.sql(f"""
        CREATE TABLE "{name}"
        ({dtypes});
        """)
    return from_native(conn.table(name))


__all__ = ["table"]

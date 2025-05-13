from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Mapping

from narwhals._spark_like.utils import narwhals_to_native_dtype
from narwhals.translate import from_native
from narwhals.utils import Version

if TYPE_CHECKING:
    from sqlframe.standalone import StandaloneDataFrame

    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.schema import Schema


def table(
    name: str, schema: Mapping[str, DType | type[DType]] | Schema
) -> LazyFrame[StandaloneDataFrame]:
    """Generate standalone LazyFrame which you can use to generate SQL.

    Note that this requires SQLFrame to be installed.

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
    try:
        from sqlframe.standalone import StandaloneSession
    except ImportError as exc:  # pragma: no cover
        msg = (
            "`narwhals.sql` requires SQLFrame to be installed.\n\n"
            "Hint: `pip install -U sqlframe"
        )
        raise ModuleNotFoundError(msg) from exc
    from sqlframe.standalone import types

    session = StandaloneSession.builder.getOrCreate()
    session.catalog.add_table(
        name,
        column_mapping={
            col: narwhals_to_native_dtype(dtype, Version.MAIN, types)
            for col, dtype in schema.items()
        },
    )
    return from_native(session.read.table(name))


__all__ = ["table"]

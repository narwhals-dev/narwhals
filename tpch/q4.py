# ruff: noqa
from datetime import datetime
from typing import Any

import polars

from narwhals import to_original_object
from narwhals import to_polars_api

Q_NUM = 4


def q4(
    lineitem_ds_raw: Any,
    orders_ds_raw: Any,
):
    var_1 = datetime(1993, 7, 1)
    var_2 = datetime(1993, 10, 1)

    line_item_ds, pl = to_polars_api(lineitem_ds_raw, version="0.20")
    orders_ds, _ = to_polars_api(orders_ds_raw, version="0.20")

    result = (
        line_item_ds.join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .filter(pl.col("o_orderdate").is_between(var_1, var_2, closed="left"))
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .unique(subset=["o_orderpriority", "l_orderkey"])
        .group_by("o_orderpriority")
        .agg(pl.len().alias("order_count"))
        .sort(by="o_orderpriority")
        .with_columns(
            pl.col("order_count")
            # .cast(pl.datatypes.Int64)
        )
    )

    return to_original_object(result.collect())


lineitem_ds = polars.scan_parquet("../tpch-data/lineitem.parquet")
orders_ds = polars.scan_parquet("../tpch-data/orders.parquet")
print(
    q4(
        lineitem_ds.collect().to_pandas(),
        orders_ds.collect().to_pandas(),
    )
)
print(
    q4(
        lineitem_ds,
        orders_ds,
    )
)

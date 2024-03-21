# ruff: noqa
from datetime import datetime
from typing import Any

import polars

import narwhals as nw

Q_NUM = 4


def q4(
    lineitem_ds_raw: Any,
    orders_ds_raw: Any,
) -> Any:
    var_1 = datetime(1993, 7, 1)
    var_2 = datetime(1993, 10, 1)

    line_item_ds = nw.LazyFrame(lineitem_ds_raw)
    orders_ds = nw.LazyFrame(orders_ds_raw)

    result = (
        line_item_ds.join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .filter(nw.col("o_orderdate").is_between(var_1, var_2, closed="left"))
        .filter(nw.col("l_commitdate") < nw.col("l_receiptdate"))
        .unique(subset=["o_orderpriority", "l_orderkey"])
        .group_by("o_orderpriority")
        .agg(nw.len().alias("order_count"))
        .sort(by="o_orderpriority")
        .with_columns(nw.col("order_count").cast(nw.Int64))
    )

    return nw.to_native(result.collect())


lineitem_ds = polars.scan_parquet("../tpch-data/s1/lineitem.parquet")
orders_ds = polars.scan_parquet("../tpch-data/s1/orders.parquet")
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

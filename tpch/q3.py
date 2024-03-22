# ruff: noqa
from typing import Any

from datetime import datetime

import polars
import pandas as pd

import narwhals as nw
import polars

polars.Config.set_tbl_cols(10)
pd.set_option("display.max_columns", 10)


def q3(
    customer_ds_raw: Any,
    line_item_ds_raw: Any,
    orders_ds_raw: Any,
) -> Any:
    var_1 = var_2 = datetime(1995, 3, 15)
    var_3 = "BUILDING"

    customer_ds = nw.from_native(customer_ds_raw)
    line_item_ds = nw.from_native(line_item_ds_raw)
    orders_ds = nw.from_native(orders_ds_raw)

    q_final = (
        customer_ds.filter(nw.col("c_mktsegment") == var_3)
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .filter(nw.col("o_orderdate") < var_2)
        .filter(nw.col("l_shipdate") > var_1)
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue")
        )
        .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
        .agg([nw.sum("revenue")])
        .select(
            [
                nw.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            ]
        )
        .sort(by=["revenue", "o_orderdate"], descending=[True, False])
        .head(10)
    )

    return nw.to_native(q_final)


customer_ds = polars.scan_parquet("../tpch-data/s1/customer.parquet")
lineitem_ds = polars.scan_parquet("../tpch-data/s1/lineitem.parquet")
orders_ds = polars.scan_parquet("../tpch-data/s1/orders.parquet")
print(
    q3(
        customer_ds.collect().to_pandas(),
        lineitem_ds.collect().to_pandas(),
        orders_ds.collect().to_pandas(),
    )
)
print(
    q3(
        customer_ds,
        lineitem_ds,
        orders_ds,
    ).collect()
)

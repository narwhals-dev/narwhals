# ruff: noqa
from datetime import datetime
from typing import Any

import polars

from narwhals import translate_frame


def q5(
    region_ds_raw: Any,
    nation_ds_raw: Any,
    customer_ds_raw: Any,
    lineitem_ds_raw: Any,
    orders_ds_raw: Any,
    supplier_ds_raw: Any,
) -> Any:
    var_1 = "ASIA"
    var_2 = datetime(1994, 1, 1)
    var_3 = datetime(1995, 1, 1)

    region_ds, pl = translate_frame(region_ds_raw, is_lazy=True)
    nation_ds, _ = translate_frame(nation_ds_raw, is_lazy=True)
    customer_ds, _ = translate_frame(customer_ds_raw, is_lazy=True)
    line_item_ds, _ = translate_frame(lineitem_ds_raw, is_lazy=True)
    orders_ds, _ = translate_frame(orders_ds_raw, is_lazy=True)
    supplier_ds, _ = translate_frame(supplier_ds_raw, is_lazy=True)

    result = (
        region_ds.join(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier_ds,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(pl.col("r_name") == var_1)
        .filter(pl.col("o_orderdate").is_between(var_2, var_3, closed="left"))
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg([pl.sum("revenue")])
        .sort(by="revenue", descending=True)
    )

    return result.collect().to_native()


region_ds = polars.scan_parquet("../tpch-data/region.parquet")
nation_ds = polars.scan_parquet("../tpch-data/nation.parquet")
customer_ds = polars.scan_parquet("../tpch-data/customer.parquet")
lineitem_ds = polars.scan_parquet("../tpch-data/lineitem.parquet")
orders_ds = polars.scan_parquet("../tpch-data/orders.parquet")
supplier_ds = polars.scan_parquet("../tpch-data/supplier.parquet")
print(
    q5(
        region_ds.collect().to_pandas(),
        nation_ds.collect().to_pandas(),
        customer_ds.collect().to_pandas(),
        lineitem_ds.collect().to_pandas(),
        orders_ds.collect().to_pandas(),
        supplier_ds.collect().to_pandas(),
    )
)
print(
    q5(
        region_ds,
        nation_ds,
        customer_ds,
        lineitem_ds,
        orders_ds,
        supplier_ds,
    )
)

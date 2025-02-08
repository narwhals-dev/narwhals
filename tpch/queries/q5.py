from __future__ import annotations

from datetime import datetime

import narwhals as nw


def query(
    region_ds: nw.LazyFrame,
    nation_ds: nw.LazyFrame,
    customer_ds: nw.LazyFrame,
    line_item_ds: nw.LazyFrame,
    orders_ds: nw.LazyFrame,
    supplier_ds: nw.LazyFrame,
) -> nw.LazyFrame:
    var_1 = "ASIA"
    var_2 = datetime(1994, 1, 1)
    var_3 = datetime(1995, 1, 1)

    return (
        region_ds.join(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier_ds,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(
            nw.col("r_name") == var_1,
            nw.col("o_orderdate").is_between(var_2, var_3, closed="left"),
        )
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg([nw.sum("revenue")])
        .sort(by="revenue", descending=True)
    )

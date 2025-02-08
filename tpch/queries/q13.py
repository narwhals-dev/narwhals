from __future__ import annotations

import narwhals as nw


def query(customer_ds: nw.LazyFrame, orders_ds: nw.LazyFrame) -> nw.LazyFrame:
    var1 = "special"
    var2 = "requests"

    orders = orders_ds.filter(~nw.col("o_comment").str.contains(f"{var1}.*{var2}"))
    return (
        customer_ds.join(orders, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey")
        .agg(nw.col("o_orderkey").count().alias("c_count"))
        .group_by("c_count")
        .agg(nw.len())
        .select(nw.col("c_count"), nw.col("len").alias("custdist"))
        .sort(by=["custdist", "c_count"], descending=[True, True])
    )

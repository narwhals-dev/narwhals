from __future__ import annotations

from datetime import datetime

import narwhals as nw


def query(
    customer_ds: nw.LazyFrame,
    nation_ds: nw.LazyFrame,
    lineitem_ds: nw.LazyFrame,
    orders_ds: nw.LazyFrame,
) -> nw.LazyFrame:
    var1 = datetime(1993, 10, 1)
    var2 = datetime(1994, 1, 1)

    return (
        customer_ds.join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(nation_ds, left_on="c_nationkey", right_on="n_nationkey")
        .filter(nw.col("o_orderdate").is_between(var1, var2, closed="left"))
        .filter(nw.col("l_returnflag") == "R")
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue")
        )
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg(nw.sum("revenue"))
        .select(
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        )
        .sort(by="revenue", descending=True)
        .head(20)
    )

from __future__ import annotations

import narwhals as nw


def query(
    part_ds: nw.LazyFrame,
    partsupp_ds: nw.LazyFrame,
    nation_ds: nw.LazyFrame,
    lineitem_ds: nw.LazyFrame,
    orders_ds: nw.LazyFrame,
    supplier_ds: nw.LazyFrame,
) -> nw.LazyFrame:
    return (
        part_ds.join(partsupp_ds, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            lineitem_ds,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .filter(nw.col("p_name").str.contains("green"))
        .select(
            nw.col("n_name").alias("nation"),
            nw.col("o_orderdate").dt.year().alias("o_year"),
            (
                nw.col("l_extendedprice") * (1 - nw.col("l_discount"))
                - nw.col("ps_supplycost") * nw.col("l_quantity")
            ).alias("amount"),
        )
        .group_by("nation", "o_year")
        .agg(nw.sum("amount").alias("sum_profit"))
        .sort(by=["nation", "o_year"], descending=[False, True])
    )

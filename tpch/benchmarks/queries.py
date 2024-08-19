from __future__ import annotations

from datetime import date

import narwhals.stable.v1 as nw


def q1(lineitem: nw.LazyFrame) -> nw.DataFrame:
    var_1 = date(1998, 9, 2)
    query_result = (
        lineitem.filter(nw.col("l_shipdate") <= var_1)
        .with_columns(
            disc_price=nw.col("l_extendedprice") * (1 - nw.col("l_discount")),
            charge=(
                nw.col("l_extendedprice")
                * (1.0 - nw.col("l_discount"))
                * (1.0 + nw.col("l_tax"))
            ),
        )
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                nw.col("l_quantity").sum().alias("sum_qty"),
                nw.col("l_extendedprice").sum().alias("sum_base_price"),
                nw.col("disc_price").sum().alias("sum_disc_price"),
                nw.col("charge").sum().alias("sum_charge"),
                nw.col("l_quantity").mean().alias("avg_qty"),
                nw.col("l_extendedprice").mean().alias("avg_price"),
                nw.col("l_discount").mean().alias("avg_disc"),
                nw.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    return query_result.collect()


def q2(
    region: nw.LazyFrame,
    nation: nw.LazyFrame,
    supplier: nw.LazyFrame,
    part: nw.LazyFrame,
    part_supp: nw.LazyFrame,
) -> nw.DataFrame:
    var_1 = 15
    var_2 = "BRASS"
    var_3 = "EUROPE"

    tmp = (
        part.join(part_supp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .filter(
            nw.col("p_size") == var_1,
            nw.col("p_type").str.ends_with(var_2),
            nw.col("r_name") == var_3,
        )
    )

    final_cols = [
        "s_acctbal",
        "s_name",
        "n_name",
        "p_partkey",
        "p_mfgr",
        "s_address",
        "s_phone",
        "s_comment",
    ]

    return (
        tmp.group_by("p_partkey")
        .agg(nw.col("ps_supplycost").min().alias("ps_supplycost"))
        .join(
            tmp,
            left_on=["p_partkey", "ps_supplycost"],
            right_on=["p_partkey", "ps_supplycost"],
        )
        .select(final_cols)
        .sort(
            ["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .head(100)
        .collect()
    )


def q3(
    customer: nw.LazyFrame, line_item: nw.LazyFrame, orders: nw.LazyFrame
) -> nw.DataFrame:
    var_1 = var_2 = date(1995, 3, 15)
    var_3 = "BUILDING"

    return (
        customer.filter(nw.col("c_mktsegment") == var_3)
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(line_item, left_on="o_orderkey", right_on="l_orderkey")
        .filter(
            nw.col("o_orderdate") < var_2,
            nw.col("l_shipdate") > var_1,
        )
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
        .collect()
    )

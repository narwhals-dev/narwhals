from __future__ import annotations

import narwhals as nw


def query(customer_ds: nw.LazyFrame, orders_ds: nw.LazyFrame) -> nw.LazyFrame:
    q1 = (
        customer_ds.with_columns(nw.col("c_phone").str.slice(0, 2).alias("cntrycode"))
        .filter(nw.col("cntrycode").str.contains("13|31|23|29|30|18|17"))
        .select("c_acctbal", "c_custkey", nw.col("cntrycode").cast(nw.Int64()))
    )

    q2 = q1.filter(nw.col("c_acctbal") > 0.0).select(
        nw.col("c_acctbal").mean().alias("avg_acctbal")
    )

    q3 = (
        orders_ds.select("o_custkey")
        .unique("o_custkey")
        .with_columns(nw.col("o_custkey").alias("c_custkey"))
    )

    return (
        q1.join(q3, left_on="c_custkey", right_on="c_custkey", how="left")
        .filter(nw.col("o_custkey").is_null())
        .join(q2, how="cross")
        .filter(nw.col("c_acctbal") > nw.col("avg_acctbal"))
        .group_by("cntrycode")
        .agg(
            nw.col("c_acctbal").count().alias("numcust"),
            nw.col("c_acctbal").sum().alias("totacctbal"),
        )
        .sort("cntrycode")
    )

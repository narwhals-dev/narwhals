from __future__ import annotations

import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
def query(customer_ds: FrameT, lineitem_ds: FrameT, orders_ds: FrameT) -> FrameT:
    var1 = 300

    query1 = (
        lineitem_ds.group_by("l_orderkey")
        .agg(nw.col("l_quantity").sum().alias("sum_quantity"))
        .filter(nw.col("sum_quantity") > var1)
    )

    return (
        orders_ds.join(query1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(customer_ds, left_on="o_custkey", right_on="c_custkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(nw.col("l_quantity").sum().alias("col6"))
        .select(
            nw.col("c_name"),
            nw.col("o_custkey").alias("c_custkey"),
            nw.col("o_orderkey"),
            nw.col("o_orderdate").alias("o_orderdat"),
            nw.col("o_totalprice"),
            nw.col("col6"),
        )
        .sort(by=["o_totalprice", "o_orderdat"], descending=[True, False])
        .head(100)
    )

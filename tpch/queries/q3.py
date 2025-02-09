from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    customer_ds: FrameT,
    line_item_ds: FrameT,
    orders_ds: FrameT,
) -> FrameT:
    var_1 = var_2 = datetime(1995, 3, 15)
    var_3 = "BUILDING"

    return (
        customer_ds.filter(nw.col("c_mktsegment") == var_3)
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
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
    )

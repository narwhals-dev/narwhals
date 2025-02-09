from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    lineitem_ds: FrameT,
    supplier_ds: FrameT,
) -> FrameT:
    var1 = datetime(1996, 1, 1)
    var2 = datetime(1996, 4, 1)

    revenue = (
        lineitem_ds.filter(nw.col("l_shipdate").is_between(var1, var2, closed="left"))
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias(
                "total_revenue"
            )
        )
        .group_by("l_suppkey")
        .agg(nw.sum("total_revenue"))
        .select(nw.col("l_suppkey").alias("supplier_no"), nw.col("total_revenue"))
    )

    return (
        supplier_ds.join(revenue, left_on="s_suppkey", right_on="supplier_no")
        .filter(nw.col("total_revenue") == nw.col("total_revenue").max())
        .with_columns(nw.col("total_revenue").round(2))
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
    )

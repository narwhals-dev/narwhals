from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    part_ds: FrameT,
    partsupp_ds: FrameT,
    nation_ds: FrameT,
    lineitem_ds: FrameT,
    supplier_ds: FrameT,
) -> FrameT:
    var1 = datetime(1994, 1, 1)
    var2 = datetime(1995, 1, 1)
    var3 = "CANADA"
    var4 = "forest"

    query1 = (
        lineitem_ds.filter(nw.col("l_shipdate").is_between(var1, var2, closed="left"))
        .group_by("l_partkey", "l_suppkey")
        .agg((nw.col("l_quantity").sum()).alias("sum_quantity"))
        .with_columns(sum_quantity=nw.col("sum_quantity") * 0.5)
    )
    query2 = nation_ds.filter(nw.col("n_name") == var3)
    query3 = supplier_ds.join(query2, left_on="s_nationkey", right_on="n_nationkey")

    return (
        part_ds.filter(nw.col("p_name").str.starts_with(var4))
        .select("p_partkey")
        .unique("p_partkey")
        .join(partsupp_ds, left_on="p_partkey", right_on="ps_partkey")
        .join(
            query1,
            left_on=["ps_suppkey", "p_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        .filter(nw.col("ps_availqty") > nw.col("sum_quantity"))
        .select("ps_suppkey")
        .unique("ps_suppkey")
        .join(query3, left_on="ps_suppkey", right_on="s_suppkey")
        .select("s_name", "s_address")
        .sort("s_name")
    )

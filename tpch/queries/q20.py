from datetime import datetime

import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
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
        .select(nw.col("p_partkey").unique())
        .join(partsupp_ds, left_on="p_partkey", right_on="ps_partkey")
        .join(
            query1,
            left_on=["ps_suppkey", "p_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        .filter(nw.col("ps_availqty") > nw.col("sum_quantity"))
        .select(nw.col("ps_suppkey").unique())
        .join(query3, left_on="ps_suppkey", right_on="s_suppkey")
        .select("s_name", "s_address")
        .sort("s_name")
    )

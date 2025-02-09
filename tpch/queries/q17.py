from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(lineitem_ds: FrameT, part_ds: FrameT) -> FrameT:
    var1 = "Brand#23"
    var2 = "MED BOX"

    query1 = (
        part_ds.filter(nw.col("p_brand") == var1)
        .filter(nw.col("p_container") == var2)
        .join(lineitem_ds, how="left", left_on="p_partkey", right_on="l_partkey")
    )

    return (
        query1.with_columns(l_quantity_times_point_2=nw.col("l_quantity") * 0.2)
        .group_by("p_partkey")
        .agg(nw.col("l_quantity_times_point_2").mean().alias("avg_quantity"))
        .select(nw.col("p_partkey").alias("key"), nw.col("avg_quantity"))
        .join(query1, left_on="key", right_on="p_partkey")
        .filter(nw.col("l_quantity") < nw.col("avg_quantity"))
        .select((nw.col("l_extendedprice").sum() / 7.0).round(2).alias("avg_yearly"))
    )

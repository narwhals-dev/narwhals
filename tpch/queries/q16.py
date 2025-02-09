from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(part_ds: FrameT, partsupp_ds: FrameT, supplier_ds: FrameT) -> FrameT:
    var1 = "Brand#45"

    supplier = supplier_ds.filter(
        nw.col("s_comment").str.contains(".*Customer.*Complaints.*")
    ).select(nw.col("s_suppkey"), nw.col("s_suppkey").alias("ps_suppkey"))

    return (
        part_ds.join(partsupp_ds, left_on="p_partkey", right_on="ps_partkey")
        .filter(nw.col("p_brand") != var1)
        .filter(~nw.col("p_type").str.contains("MEDIUM POLISHED*"))
        .filter(nw.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey", how="left")
        .filter(nw.col("ps_suppkey_right").is_null())
        .group_by("p_brand", "p_type", "p_size")
        .agg(nw.col("ps_suppkey").n_unique().alias("supplier_cnt"))
        .sort(
            by=["supplier_cnt", "p_brand", "p_type", "p_size"],
            descending=[True, False, False, False],
        )
    )

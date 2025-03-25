from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    part_ds: FrameT,
    supplier_ds: FrameT,
    line_item_ds: FrameT,
    orders_ds: FrameT,
    customer_ds: FrameT,
    nation_ds: FrameT,
    region_ds: FrameT,
) -> FrameT:
    nation = "BRAZIL"
    region = "AMERICA"
    type = "ECONOMY ANODIZED STEEL"
    date1 = date(1995, 1, 1)
    date2 = date(1996, 12, 31)

    n1 = nation_ds.select("n_nationkey", "n_regionkey")
    n2 = nation_ds.select("n_nationkey", "n_name")

    return (
        part_ds.join(line_item_ds, left_on="p_partkey", right_on="l_partkey")
        .join(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        .join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer_ds, left_on="o_custkey", right_on="c_custkey")
        .join(n1, left_on="c_nationkey", right_on="n_nationkey")
        .join(region_ds, left_on="n_regionkey", right_on="r_regionkey")
        .filter(nw.col("r_name") == region)
        .join(n2, left_on="s_nationkey", right_on="n_nationkey")
        .filter(nw.col("o_orderdate").is_between(date1, date2))
        .filter(nw.col("p_type") == type)
        .select(
            nw.col("o_orderdate").dt.year().alias("o_year"),
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("volume"),
            nw.col("n_name").alias("nation"),
        )
        .with_columns(
            nw.when(nw.col("nation") == nation)
            .then(nw.col("volume"))
            .otherwise(0)
            .alias("_tmp")
        )
        .group_by("o_year")
        .agg(_tmp_sum=nw.sum("_tmp"), volume_sum=nw.sum("volume"))
        .select("o_year", mkt_share=nw.col("_tmp_sum") / nw.col("volume_sum"))
        .sort("o_year")
    )

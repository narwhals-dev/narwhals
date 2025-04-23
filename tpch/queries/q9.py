from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    part_ds: FrameT,
    partsupp_ds: FrameT,
    nation_ds: FrameT,
    lineitem_ds: FrameT,
    orders_ds: FrameT,
    supplier_ds: FrameT,
) -> FrameT:
    return (
        part_ds.join(partsupp_ds, left_on="p_partkey", right_on="ps_partkey")  # pyright: ignore[reportArgumentType]
        .join(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")  # pyright: ignore[reportArgumentType]
        .join(
            lineitem_ds,  # pyright: ignore[reportArgumentType]
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")  # pyright: ignore[reportArgumentType]
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")  # pyright: ignore[reportArgumentType]
        .filter(nw.col("p_name").str.contains("green"))
        .select(
            nw.col("n_name").alias("nation"),
            nw.col("o_orderdate").dt.year().alias("o_year"),
            (
                nw.col("l_extendedprice") * (1 - nw.col("l_discount"))
                - nw.col("ps_supplycost") * nw.col("l_quantity")
            ).alias("amount"),
        )
        .group_by("nation", "o_year")
        .agg(nw.sum("amount").alias("sum_profit"))
        .sort(by=["nation", "o_year"], descending=[False, True])
    )

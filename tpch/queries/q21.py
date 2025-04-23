from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    lineitem: FrameT,
    nation: FrameT,
    orders: FrameT,
    supplier: FrameT,
) -> FrameT:
    var1 = "SAUDI ARABIA"

    q1 = (
        lineitem.group_by("l_orderkey")
        .agg(nw.len().alias("n_supp_by_order"))
        .filter(nw.col("n_supp_by_order") > 1)
        .join(
            lineitem.filter(nw.col("l_receiptdate") > nw.col("l_commitdate")),  # pyright: ignore[reportArgumentType]
            left_on="l_orderkey",
            right_on="l_orderkey",
        )
    )

    return (
        q1.group_by("l_orderkey")
        .agg(nw.len().alias("n_supp_by_order"))
        .join(
            q1,  # pyright: ignore[reportArgumentType]
            left_on="l_orderkey",
            right_on="l_orderkey",
        )
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")  # pyright: ignore[reportArgumentType]
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")  # pyright: ignore[reportArgumentType]
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")  # pyright: ignore[reportArgumentType]
        .filter(nw.col("n_supp_by_order") == 1)
        .filter(nw.col("n_name") == var1)
        .filter(nw.col("o_orderstatus") == "F")
        .group_by("s_name")
        .agg(nw.len().alias("numwait"))
        .sort(by=["numwait", "s_name"], descending=[True, False])
        .head(100)
    )

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(line_item_ds: FrameT, orders_ds: FrameT) -> FrameT:
    var_1 = datetime(1993, 7, 1)
    var_2 = datetime(1993, 10, 1)

    return (
        line_item_ds.join(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        .filter(
            nw.col("o_orderdate").is_between(var_1, var_2, closed="left"),
            nw.col("l_commitdate") < nw.col("l_receiptdate"),
        )
        .unique(subset=["o_orderpriority", "l_orderkey"])
        .group_by("o_orderpriority")
        .agg(nw.len().alias("order_count"))
        .sort(by="o_orderpriority")
        .with_columns(nw.col("order_count").cast(nw.Int64))
    )

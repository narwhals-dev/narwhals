from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(lineitem: FrameT) -> FrameT:
    var_1 = datetime(1998, 9, 2)
    return (
        lineitem.filter(nw.col("l_shipdate") <= var_1)
        .with_columns(
            disc_price=nw.col("l_extendedprice") * (1 - nw.col("l_discount")),
            charge=(
                nw.col("l_extendedprice")
                * (1.0 - nw.col("l_discount"))
                * (1.0 + nw.col("l_tax"))
            ),
        )
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            nw.sum("l_quantity").alias("sum_qty"),
            nw.sum("l_extendedprice").alias("sum_base_price"),
            nw.sum("disc_price").alias("sum_disc_price"),
            nw.sum("charge").alias("sum_charge"),
            nw.mean("l_quantity").alias("avg_qty"),
            nw.mean("l_extendedprice").alias("avg_price"),
            nw.mean("l_discount").alias("avg_disc"),
            nw.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )

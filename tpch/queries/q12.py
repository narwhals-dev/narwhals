from __future__ import annotations

from datetime import datetime

import narwhals as nw


def query(line_item_ds: nw.LazyFrame, orders_ds: nw.LazyFrame) -> nw.LazyFrame:
    var1 = "MAIL"
    var2 = "SHIP"
    var3 = datetime(1994, 1, 1)
    var4 = datetime(1995, 1, 1)

    return (
        orders_ds.join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .filter(nw.col("l_shipmode").is_in([var1, var2]))
        .filter(nw.col("l_commitdate") < nw.col("l_receiptdate"))
        .filter(nw.col("l_shipdate") < nw.col("l_commitdate"))
        .filter(nw.col("l_receiptdate").is_between(var3, var4, closed="left"))
        .with_columns(
            nw.when(nw.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]))
            .then(1)
            .otherwise(0)
            .alias("high_line_count"),
            nw.when(~nw.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]))
            .then(1)
            .otherwise(0)
            .alias("low_line_count"),
        )
        .group_by("l_shipmode")
        .agg(nw.col("high_line_count").sum(), nw.col("low_line_count").sum())
        .sort("l_shipmode")
    )

from datetime import datetime

import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
def query(
    region_ds_raw: FrameT,
    nation_ds_raw: FrameT,
    customer_ds_raw: FrameT,
    line_item_ds_raw: FrameT,
    orders_ds_raw: FrameT,
    supplier_ds_raw: FrameT,
) -> FrameT:
    var_1 = "ASIA"
    var_2 = datetime(1994, 1, 1)
    var_3 = datetime(1995, 1, 1)

    region_ds = nw.from_native(region_ds_raw)
    nation_ds = nw.from_native(nation_ds_raw)
    customer_ds = nw.from_native(customer_ds_raw)
    line_item_ds = nw.from_native(line_item_ds_raw)
    orders_ds = nw.from_native(orders_ds_raw)
    supplier_ds = nw.from_native(supplier_ds_raw)

    result = (
        region_ds.join(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier_ds,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(
            nw.col("r_name") == var_1,
            nw.col("o_orderdate").is_between(var_2, var_3, closed="left"),
        )
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg([nw.sum("revenue")])
        .sort(by="revenue", descending=True)
    )

    return nw.to_native(result)

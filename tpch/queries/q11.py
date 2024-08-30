from datetime import datetime

import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
def query(
    nation_ds_raw: FrameT,
    partsupp_ds_raw: FrameT,
    supplier_ds_raw: FrameT,
) -> FrameT:
    var1 = datetime(1993, 10, 1)
    var2 = datetime(1994, 1, 1)

    nation_ds = nw.from_native(nation_ds_raw)
    partsupp_ds = nw.from_native(partsupp_ds_raw)
    supplier_ds = nw.from_native(supplier_ds_raw)

    var1 = "GERMANY"
    var2 = 0.0001

    q1 = (
        partsupp_ds.join(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .filter(nw.col("n_name") == var1)
    )
    q2 = q1.select(
        (nw.col("ps_supplycost") * nw.col("ps_availqty")).sum().round(2).alias("tmp")
        * var2
    )

    q_final = (
        q1.with_columns((nw.col("ps_supplycost") * nw.col("ps_availqty")).alias("value"))
        .group_by("ps_partkey")
        .agg(nw.sum("value"))
        .join(q2, how="cross")
        .filter(nw.col("value") > nw.col("tmp"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )

    return nw.to_native(q_final)

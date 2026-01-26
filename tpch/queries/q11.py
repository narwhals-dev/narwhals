from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tpch.typing_ import todo_mark

if TYPE_CHECKING:
    from narwhals.typing import FrameT


@todo_mark(
    "Can fail when `scale_factor=0.11`.",
    "Seems to be an upstream issue for hardcoded `var2 = 0.0001`\n"
    "https://github.com/duckdb/duckdb/issues/17965",
)
def query(nation_ds: FrameT, partsupp_ds: FrameT, supplier_ds: FrameT) -> FrameT:
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

    return (
        q1.with_columns((nw.col("ps_supplycost") * nw.col("ps_availqty")).alias("value"))
        .group_by("ps_partkey")
        .agg(nw.sum("value"))
        .join(q2, how="cross")  # pyright: ignore[reportArgumentType]
        .filter(nw.col("value") > nw.col("tmp"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )

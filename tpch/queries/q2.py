from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(
    region_ds: FrameT,
    nation_ds: FrameT,
    supplier_ds: FrameT,
    part_ds: FrameT,
    part_supp_ds: FrameT,
) -> FrameT:
    var_1 = 15
    var_2 = "BRASS"
    var_3 = "EUROPE"

    result_q2 = (
        part_ds.join(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .join(region_ds, left_on="n_regionkey", right_on="r_regionkey")
        .filter(
            nw.col("p_size") == var_1,
            nw.col("p_type").str.ends_with(var_2),
            nw.col("r_name") == var_3,
        )
    )

    final_cols = [
        "s_acctbal",
        "s_name",
        "n_name",
        "p_partkey",
        "p_mfgr",
        "s_address",
        "s_phone",
        "s_comment",
    ]

    return (
        result_q2.group_by("p_partkey")
        .agg(nw.col("ps_supplycost").min().alias("ps_supplycost"))
        .join(
            result_q2,
            left_on=["p_partkey", "ps_supplycost"],
            right_on=["p_partkey", "ps_supplycost"],
        )
        .select(final_cols)
        .sort(
            ["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .head(100)
    )

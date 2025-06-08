from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import FrameT


def query(line_item_ds: FrameT) -> FrameT:
    var_1 = datetime(1994, 1, 1)
    var_2 = datetime(1995, 1, 1)
    var_3 = 24

    return (
        line_item_ds.filter(
            nw.col("l_shipdate").is_between(var_1, var_2, closed="left"),
            nw.col("l_discount").is_between(0.05, 0.07),
            nw.col("l_quantity") < var_3,
        )
        .with_columns((nw.col("l_extendedprice") * nw.col("l_discount")).alias("revenue"))
        .select(nw.sum("revenue"))
    )

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import pytest

import narwhals.stable.v1 as nw


def q1(lineitem_ds: Any) -> Any:
    var_1 = date(1998, 9, 2)
    query_result = (
        lineitem_ds.filter(nw.col("l_shipdate") <= var_1)
        .with_columns(
            disc_price=nw.col("l_extendedprice") * (1 - nw.col("l_discount")),
            charge=(
                nw.col("l_extendedprice")
                * (1.0 - nw.col("l_discount"))
                * (1.0 + nw.col("l_tax"))
            ),
        )
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                nw.col("l_quantity").sum().alias("sum_qty"),
                nw.col("l_extendedprice").sum().alias("sum_base_price"),
                nw.col("disc_price").sum().alias("sum_disc_price"),
                nw.col("charge").sum().alias("sum_charge"),
                nw.col("l_quantity").mean().alias("avg_qty"),
                nw.col("l_extendedprice").mean().alias("avg_price"),
                nw.col("l_discount").mean().alias("avg_disc"),
                nw.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    return query_result.collect()


@pytest.mark.parametrize("library", ["polars", "pyarrow"])
def test_q1(benchmark: Any, library: str) -> None:
    lib_to_reader = {
        "polars": pl.scan_parquet,
        "pyarrow": pq.read_table,
    }

    read_fn = lib_to_reader[library]
    lineitem_ds = nw.from_native(read_fn("tests/data/lineitem.parquet")).lazy()

    _ = benchmark(q1, lineitem_ds)

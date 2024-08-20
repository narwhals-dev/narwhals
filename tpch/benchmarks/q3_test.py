from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw
from tpch.benchmarks.utils import lib_to_reader

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture

DATA_FOLDER = Path("tests/data")


@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
def test_q3(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    customer = nw.from_native(read_fn(DATA_FOLDER / "customer.parquet")).lazy()
    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()

    _ = benchmark(q3, customer, lineitem, orders)


def q3(
    customer: nw.LazyFrame, line_item: nw.LazyFrame, orders: nw.LazyFrame
) -> nw.DataFrame:
    var_1 = var_2 = date(1995, 3, 15)
    var_3 = "BUILDING"

    return (
        customer.filter(nw.col("c_mktsegment") == var_3)
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(line_item, left_on="o_orderkey", right_on="l_orderkey")
        .filter(
            nw.col("o_orderdate") < var_2,
            nw.col("l_shipdate") > var_1,
        )
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue")
        )
        .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
        .agg([nw.sum("revenue")])
        .select(
            [
                nw.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            ]
        )
        .sort(by=["revenue", "o_orderdate"], descending=[True, False])
        .head(10)
        .collect()
    )

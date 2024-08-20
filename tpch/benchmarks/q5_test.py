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
def test_q5(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    customer = nw.from_native(read_fn(DATA_FOLDER / "customer.parquet")).lazy()
    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()
    region = nw.from_native(read_fn(DATA_FOLDER / "region.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q5, region, nation, customer, lineitem, orders, supplier)


def q5(
    region: nw.LazyFrame,
    nation: nw.LazyFrame,
    customer: nw.LazyFrame,
    lineitem: nw.LazyFrame,
    orders: nw.LazyFrame,
    supplier: nw.LazyFrame,
) -> nw.DataFrame:
    var_1 = "ASIA"
    var_2 = date(1994, 1, 1)
    var_3 = date(1995, 1, 1)

    return (
        region.join(nation, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier,
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
        .collect()
    )

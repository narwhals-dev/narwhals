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
def test_q8(benchmark: BenchmarkFixture, library: str) -> None:
    # Requires nw.when to be implemented first
    return
    read_fn = lib_to_reader[library]

    customer = nw.from_native(read_fn(DATA_FOLDER / "customer.parquet")).lazy()
    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()
    part = nw.from_native(read_fn(DATA_FOLDER / "part.parquet")).lazy()
    region = nw.from_native(read_fn(DATA_FOLDER / "region.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q8, nation, customer, lineitem, orders, supplier, part, region)


def q8(
    nation: nw.LazyFrame,
    customer: nw.LazyFrame,
    lineitem: nw.LazyFrame,
    orders: nw.LazyFrame,
    supplier: nw.LazyFrame,
    part: nw.LazyFrame,
    region: nw.LazyFrame,
) -> None:
    n1 = nation.select("n_nationkey", "n_regionkey")
    n2 = nation.select("n_nationkey", "n_name")

    return (
        part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(n1, left_on="c_nationkey", right_on="n_nationkey")
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .filter(nw.col("r_name") == "AMERICA")
        .join(n2, left_on="s_nationkey", right_on="n_nationkey")
        .filter(
            nw.col("o_orderdate") >= date(1995, 1, 1),
            nw.col("o_orderdate") <= date(1996, 12, 31),
        )
        .filter(nw.col("p_type") == "ECONOMY ANODIZED STEEL")
        .select(
            nw.col("o_orderdate").dt.year().alias("o_year"),
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("volume"),
            nw.col("n_name").alias("nation"),
        )
        .with_columns(
            nw.when(nw.col("nation") == "BRAZIL")
            .then(nw.col("volume"))
            .otherwise(0)
            .alias("_tmp")
        )
        .group_by("o_year")
        .agg((nw.sum("_tmp") / nw.sum("volume")).round(2).alias("mkt_share"))
        .sort("o_year")
        .collect()
    )

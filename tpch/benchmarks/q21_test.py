from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw
from tpch.benchmarks.utils import lib_to_reader

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture

DATA_FOLDER = Path("tests/data")


@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
def test_q21(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q21, lineitem, nation, orders, supplier)


def q21(
    lineitem: nw.LazyFrame,
    nation: nw.LazyFrame,
    orders: nw.LazyFrame,
    supplier: nw.LazyFrame,
) -> nw.DataFrame:
    var1 = "SAUDI ARABIA"

    q1 = (
        lineitem.group_by("l_orderkey")
        .agg(nw.len().alias("n_supp_by_order"))
        .filter(nw.col("n_supp_by_order") > 1)
        .join(
            lineitem.filter(nw.col("l_receiptdate") > nw.col("l_commitdate")),
            left_on="l_orderkey",
            right_on="l_orderkey",
        )
    )

    return (
        q1.group_by("l_orderkey")
        .agg(nw.len().alias("n_supp_by_order"))
        .join(q1, left_on="l_orderkey", right_on="l_orderkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .filter(nw.col("n_supp_by_order") == 1)
        .filter(nw.col("n_name") == var1)
        .filter(nw.col("o_orderstatus") == "F")
        .group_by("s_name")
        .agg(nw.len().alias("numwait"))
        .sort(by=["numwait", "s_name"], descending=[True, False])
        .head(100)
        .collect()
    )

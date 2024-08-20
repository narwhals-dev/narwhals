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
def test_q18(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    customer = nw.from_native(read_fn(DATA_FOLDER / "customer.parquet")).lazy()
    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()

    _ = benchmark(q18, customer, lineitem, orders)


def q18(
    customer: nw.LazyFrame, lineitem: nw.LazyFrame, orders: nw.LazyFrame
) -> nw.DataFrame:
    var1 = 300

    query1 = (
        lineitem.group_by("l_orderkey")
        .agg(nw.col("l_quantity").sum().alias("sum_quantity"))
        .filter(nw.col("sum_quantity") > var1)
    )

    return (
        orders.join(query1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(nw.col("l_quantity").sum().alias("col6"))
        .select(
            nw.col("c_name"),
            nw.col("o_custkey").alias("c_custkey"),
            nw.col("o_orderkey"),
            nw.col("o_orderdate").alias("o_orderdat"),
            nw.col("o_totalprice"),
            nw.col("col6"),
        )
        .sort(by=["o_totalprice", "o_orderdat"], descending=[True, False])
        .head(100)
        .collect()
    )

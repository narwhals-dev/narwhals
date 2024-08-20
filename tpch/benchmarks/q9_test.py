from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tpch.benchmarks.utils import lib_to_reader

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture

DATA_FOLDER = Path("tests/data")


@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
def test_q9(benchmark: BenchmarkFixture, library: str, request: Any) -> None:
    if library == "dask":
        # Requires cast method
        request.applymarker(pytest.mark.xfail)
    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()
    part = nw.from_native(read_fn(DATA_FOLDER / "part.parquet")).lazy()
    partsupp = nw.from_native(read_fn(DATA_FOLDER / "partsupp.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q9, part, partsupp, nation, lineitem, orders, supplier)


def q9(
    part: nw.LazyFrame,
    partsupp: nw.LazyFrame,
    nation: nw.LazyFrame,
    lineitem: nw.LazyFrame,
    orders: nw.LazyFrame,
    supplier: nw.LazyFrame,
) -> nw.DataFrame:
    return (
        part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            lineitem,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .filter(nw.col("p_name").str.contains("green"))
        .select(
            nw.col("n_name").alias("nation"),
            nw.col("o_orderdate").cast(nw.Datetime).dt.year().alias("o_year"),
            (
                nw.col("l_extendedprice") * (1 - nw.col("l_discount"))
                - nw.col("ps_supplycost") * nw.col("l_quantity")
            ).alias("amount"),
        )
        .group_by("nation", "o_year")
        .agg(nw.sum("amount").alias("sum_profit"))
        .sort(by=["nation", "o_year"], descending=[False, True])
        .collect()
    )

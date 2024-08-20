from __future__ import annotations

from datetime import date
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
def test_q20(benchmark: BenchmarkFixture, library: str, request: Any) -> None:
    if library == "dask":
        # requires unique
        request.applymarker(pytest.mark.xfail)
    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    part = nw.from_native(read_fn(DATA_FOLDER / "part.parquet")).lazy()
    partsupp = nw.from_native(read_fn(DATA_FOLDER / "partsupp.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q20, part, partsupp, nation, lineitem, supplier)


def q20(
    part: nw.LazyFrame,
    partsupp: nw.LazyFrame,
    nation: nw.LazyFrame,
    lineitem: nw.LazyFrame,
    supplier: nw.LazyFrame,
) -> nw.DataFrame:
    var1 = date(1994, 1, 1)
    var2 = date(1995, 1, 1)
    var3 = "CANADA"
    var4 = "forest"

    query1 = (
        lineitem.filter(nw.col("l_shipdate").is_between(var1, var2, closed="left"))
        .group_by("l_partkey", "l_suppkey")
        .agg((nw.col("l_quantity").sum()).alias("sum_quantity"))
        .with_columns(sum_quantity=nw.col("sum_quantity") * 0.5)
    )
    query2 = nation.filter(nw.col("n_name") == var3)
    query3 = supplier.join(query2, left_on="s_nationkey", right_on="n_nationkey")

    return (
        part.filter(nw.col("p_name").str.starts_with(var4))
        .select(nw.col("p_partkey").unique())
        .join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(
            query1,
            left_on=["ps_suppkey", "p_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        .filter(nw.col("ps_availqty") > nw.col("sum_quantity"))
        .select(nw.col("ps_suppkey").unique())
        .join(query3, left_on="ps_suppkey", right_on="s_suppkey")
        .select("s_name", "s_address")
        .sort("s_name")
        .collect()
    )

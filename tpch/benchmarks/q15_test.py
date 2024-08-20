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
def test_q15(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q15, lineitem, supplier)


def q15(
    lineitem: nw.LazyFrame,
    supplier: nw.LazyFrame,
) -> nw.DataFrame:
    var1 = date(1996, 1, 1)
    var2 = date(1996, 4, 1)

    revenue = (
        lineitem.filter(nw.col("l_shipdate").is_between(var1, var2, closed="left"))
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias(
                "total_revenue"
            )
        )
        .group_by("l_suppkey")
        .agg(nw.sum("total_revenue"))
        .select(nw.col("l_suppkey").alias("supplier_no"), nw.col("total_revenue"))
    )

    return (
        supplier.join(revenue, left_on="s_suppkey", right_on="supplier_no")
        .filter(nw.col("total_revenue") == nw.col("total_revenue").max())
        .with_columns(nw.col("total_revenue").round(2))
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
        .collect()
    )

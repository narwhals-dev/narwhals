from __future__ import annotations

from datetime import datetime
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
def test_q7(benchmark: BenchmarkFixture, library: str, request: Any) -> None:
    if library == "dask":
        # Dasknamespace does not implement concat
        request.applymarker(pytest.mark.xfail)
    read_fn = lib_to_reader[library]

    customer = nw.from_native(read_fn(DATA_FOLDER / "customer.parquet")).lazy()
    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q7, nation, customer, lineitem, orders, supplier)


def q7(
    nation: nw.LazyFrame,
    customer: nw.LazyFrame,
    lineitem: nw.LazyFrame,
    orders: nw.LazyFrame,
    supplier: nw.LazyFrame,
) -> nw.DataFrame:
    n1 = nation.filter(nw.col("n_name") == "FRANCE")
    n2 = nation.filter(nw.col("n_name") == "GERMANY")

    var_1 = datetime(1995, 1, 1)
    var_2 = datetime(1996, 12, 31)

    df1 = (
        customer.join(n1, left_on="c_nationkey", right_on="n_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .rename({"n_name": "cust_nation"})
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(n2, left_on="s_nationkey", right_on="n_nationkey")
        .rename({"n_name": "supp_nation"})
    )

    df2 = (
        customer.join(n2, left_on="c_nationkey", right_on="n_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .rename({"n_name": "cust_nation"})
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(n1, left_on="s_nationkey", right_on="n_nationkey")
        .rename({"n_name": "supp_nation"})
    )

    return (
        nw.concat([df1, df2])
        .filter(nw.col("l_shipdate").cast(nw.Datetime).is_between(var_1, var_2))
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("volume")
        )
        .with_columns(nw.col("l_shipdate").cast(nw.Datetime).dt.year().alias("l_year"))
        .group_by("supp_nation", "cust_nation", "l_year")
        .agg(nw.sum("volume").alias("revenue"))
        .sort(by=["supp_nation", "cust_nation", "l_year"])
        .collect()
    )

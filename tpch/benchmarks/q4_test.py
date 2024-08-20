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
def test_q4(benchmark: BenchmarkFixture, library: str, request: Any) -> None:
    if library == "dask":
        # cast method is missing
        request.applymarker(pytest.mark.xfail)

    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()

    _ = benchmark(q4, lineitem, orders)


def q4(lineitem: nw.LazyFrame, orders: nw.LazyFrame) -> nw.DataFrame:
    var_1 = date(1993, 7, 1)
    var_2 = date(1993, 10, 1)

    return (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .filter(
            nw.col("o_orderdate").is_between(var_1, var_2, closed="left"),
            nw.col("l_commitdate") < nw.col("l_receiptdate"),
        )
        .unique(subset=["o_orderpriority", "l_orderkey"])
        .group_by("o_orderpriority")
        .agg(nw.len().alias("order_count"))
        .sort(by="o_orderpriority")
        .with_columns(nw.col("order_count").cast(nw.Int64))
        .collect()
    )

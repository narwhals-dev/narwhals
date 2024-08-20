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
def test_q11(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    partsupp = nw.from_native(read_fn(DATA_FOLDER / "partsupp.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q11, partsupp, nation, supplier)


def q11(
    partsupp: nw.LazyFrame, nation: nw.LazyFrame, supplier: nw.LazyFrame
) -> nw.DataFrame:
    var1 = "GERMANY"
    var2 = 0.0001

    q1 = (
        partsupp.join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .filter(nw.col("n_name") == var1)
    )
    q2 = q1.select(
        (nw.col("ps_supplycost") * nw.col("ps_availqty")).sum().round(2).alias("tmp")
        * var2
    )

    return (
        q1.with_columns((nw.col("ps_supplycost") * nw.col("ps_availqty")).alias("value"))
        .group_by("ps_partkey")
        .agg(nw.sum("value"))
        .join(q2, how="cross")
        .filter(nw.col("value") > nw.col("tmp"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
        .collect()
    )

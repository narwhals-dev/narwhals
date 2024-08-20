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
def test_q17(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    part = nw.from_native(read_fn(DATA_FOLDER / "part.parquet")).lazy()

    _ = benchmark(q17, lineitem, part)


def q17(lineitem: nw.LazyFrame, part: nw.LazyFrame) -> nw.DataFrame:
    var1 = "Brand#23"
    var2 = "MED BOX"

    query1 = (
        part.filter(nw.col("p_brand") == var1)
        .filter(nw.col("p_container") == var2)
        .join(lineitem, how="left", left_on="p_partkey", right_on="l_partkey")
    )

    return (
        query1.with_columns(avg_quantity=0.2 * nw.col("l_quantity"))
        .group_by("p_partkey")
        .agg(nw.col("avg_quantity").mean())
        .select(nw.col("p_partkey").alias("key"), nw.col("avg_quantity"))
        .join(query1, left_on="key", right_on="p_partkey")
        .filter(nw.col("l_quantity") < nw.col("avg_quantity"))
        .select((nw.col("l_extendedprice").sum() / 7.0).round(2).alias("avg_yearly"))
        .collect()
    )

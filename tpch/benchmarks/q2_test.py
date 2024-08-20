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
def test_q2(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    part = nw.from_native(read_fn(DATA_FOLDER / "part.parquet")).lazy()
    partsupp = nw.from_native(read_fn(DATA_FOLDER / "partsupp.parquet")).lazy()
    region = nw.from_native(read_fn(DATA_FOLDER / "region.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    _ = benchmark(q2, region, nation, supplier, part, partsupp)


def q2(
    region: nw.LazyFrame,
    nation: nw.LazyFrame,
    supplier: nw.LazyFrame,
    part: nw.LazyFrame,
    part_supp: nw.LazyFrame,
) -> nw.DataFrame:
    var_1 = 15
    var_2 = "BRASS"
    var_3 = "EUROPE"

    tmp = (
        part.join(part_supp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .filter(
            nw.col("p_size") == var_1,
            nw.col("p_type").str.ends_with(var_2),
            nw.col("r_name") == var_3,
        )
    )

    final_cols = [
        "s_acctbal",
        "s_name",
        "n_name",
        "p_partkey",
        "p_mfgr",
        "s_address",
        "s_phone",
        "s_comment",
    ]

    return (
        tmp.group_by("p_partkey")
        .agg(nw.col("ps_supplycost").min().alias("ps_supplycost"))
        .join(
            tmp,
            left_on=["p_partkey", "ps_supplycost"],
            right_on=["p_partkey", "ps_supplycost"],
        )
        .select(final_cols)
        .sort(
            ["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .head(100)
        .collect()
    )

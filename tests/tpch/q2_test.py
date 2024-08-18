from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

lib_to_reader = {
    "pandas": pd.read_parquet,
    "polars": pl.scan_parquet,
    "dask": lambda path: dd.read_parquet(path, dtype_backend="pyarrow"),
    "pyarrow": pq.read_table,
}


def q2(
    region_ds: Any,
    nation_ds: Any,
    supplier_ds: Any,
    part_ds: Any,
    part_supp_ds: Any,
) -> Any:
    var_1 = 15
    var_2 = "BRASS"
    var_3 = "EUROPE"

    tmp = (
        part_ds.join(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .join(region_ds, left_on="n_regionkey", right_on="r_regionkey")
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

    query_result = (
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
    )
    return query_result.collect()


@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
def test_q2(benchmark: Any, library: str, request: Any) -> None:
    if library == "pandas" and parse_version(pd.__version__) < (1, 5):
        request.applymarker(pytest.mark.xfail)

    read_fn = lib_to_reader[library]
    region_ds = nw.from_native(read_fn("tests/data/region.parquet")).lazy()
    nation_ds = nw.from_native(read_fn("tests/data/nation.parquet")).lazy()
    supplier_ds = nw.from_native(read_fn("tests/data/supplier.parquet")).lazy()
    part_ds = nw.from_native(read_fn("tests/data/part.parquet")).lazy()
    part_supp_ds = nw.from_native(read_fn("tests/data/partsupp.parquet")).lazy()

    args = (region_ds, nation_ds, supplier_ds, part_ds, part_supp_ds)

    _ = benchmark(q2, *args)

    # Need to create expected compare_dicts(result, q2_expected)

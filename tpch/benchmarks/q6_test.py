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
def test_q6(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()

    _ = benchmark(q6, lineitem)


def q6(lineitem: nw.LazyFrame) -> nw.DataFrame:
    var_1 = date(1994, 1, 1)
    var_2 = date(1995, 1, 1)
    var_3 = 24

    line_item_ds = nw.from_native(lineitem)

    return (
        line_item_ds.filter(
            nw.col("l_shipdate").is_between(var_1, var_2, closed="left"),
            nw.col("l_discount").is_between(0.05, 0.07),
            nw.col("l_quantity") < var_3,
        )
        .with_columns((nw.col("l_extendedprice") * nw.col("l_discount")).alias("revenue"))
        .select(nw.sum("revenue"))
        .collect()
    )

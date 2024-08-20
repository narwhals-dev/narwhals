from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw
from tpch.benchmarks.queries import q1
from tpch.benchmarks.queries import q2
from tpch.benchmarks.queries import q3
from tpch.benchmarks.utils import lib_to_reader

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture

DATA_FOLDER = Path("tests/data")


@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
def test_queries(benchmark: BenchmarkFixture, library: str) -> None:
    read_fn = lib_to_reader[library]

    customer = nw.from_native(read_fn(DATA_FOLDER / "customer.parquet")).lazy()
    lineitem = nw.from_native(read_fn(DATA_FOLDER / "lineitem.parquet")).lazy()
    nation = nw.from_native(read_fn(DATA_FOLDER / "nation.parquet")).lazy()
    orders = nw.from_native(read_fn(DATA_FOLDER / "orders.parquet")).lazy()
    part = nw.from_native(read_fn(DATA_FOLDER / "part.parquet")).lazy()
    partsupp = nw.from_native(read_fn(DATA_FOLDER / "partsupp.parquet")).lazy()
    region = nw.from_native(read_fn(DATA_FOLDER / "region.parquet")).lazy()
    supplier = nw.from_native(read_fn(DATA_FOLDER / "supplier.parquet")).lazy()

    q1_result = benchmark(q1, lineitem)  # noqa: F841
    q2_result = benchmark(q2, region, nation, supplier, part, partsupp)  # noqa: F841
    q3_result = benchmark(q3, customer, lineitem, orders)  # noqa: F841

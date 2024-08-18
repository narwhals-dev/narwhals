from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts

q1_expected = {
    "l_returnflag": ["A", "N", "N", "R"],
    "l_linestatus": ["F", "F", "O", "F"],
    "sum_qty": [2109.0, 29.0, 3682.0, 1876.0],
    "sum_base_price": [3114026.44, 39824.83, 5517101.99, 2947892.16],
    "sum_disc_price": [2954950.8082, 39028.3334, 5205468.4852, 2816542.4816999994],
    "sum_charge": [
        3092840.4194289995,
        39808.900068,
        5406966.873419,
        2935797.8313019997,
    ],
    "avg_qty": [27.75, 29.0, 25.047619047619047, 26.422535211267604],
    "avg_price": [
        40974.032105263155,
        39824.83,
        37531.30605442177,
        41519.607887323946,
    ],
    "avg_disc": [0.05039473684210526, 0.02, 0.05537414965986395, 0.04507042253521127],
    "count_order": [76, 1, 147, 71],
}


def q1(library: str) -> dict[str, list[Any]]:
    if library == "pandas":
        df_raw = pd.read_parquet("tests/data/lineitem.parquet")
        df_raw["l_shipdate"] = pd.to_datetime(df_raw["l_shipdate"])
    elif library == "polars":
        df_raw = pl.scan_parquet("tests/data/lineitem.parquet")
    elif library == "dask":
        pytest.importorskip("dask")
        pytest.importorskip("dask_expr", exc_type=ImportError)
        import dask.dataframe as dd

        df_raw = dd.read_parquet("tests/data/lineitem.parquet", dtype_backend="pyarrow")
    elif library == "pyarrow":
        df_raw = pq.read_table("tests/data/lineitem.parquet")

    var_1 = datetime(1998, 9, 2)
    df = nw.from_native(df_raw).lazy()
    query_result = (
        df.filter(nw.col("l_shipdate") <= var_1)
        .with_columns(
            disc_price=nw.col("l_extendedprice") * (1 - nw.col("l_discount")),
            charge=(
                nw.col("l_extendedprice")
                * (1.0 - nw.col("l_discount"))
                * (1.0 + nw.col("l_tax"))
            ),
        )
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                nw.col("l_quantity").sum().alias("sum_qty"),
                nw.col("l_extendedprice").sum().alias("sum_base_price"),
                nw.col("disc_price").sum().alias("sum_disc_price"),
                nw.col("charge").sum().alias("sum_charge"),
                nw.col("l_quantity").mean().alias("avg_qty"),
                nw.col("l_extendedprice").mean().alias("avg_price"),
                nw.col("l_discount").mean().alias("avg_disc"),
                nw.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    return query_result.collect().to_dict(as_series=False)


@pytest.mark.benchmark()
@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
def test_q1(benchmark: Any, library: str, request: Any) -> None:
    if library == "pandas" and parse_version(pd.__version__) < (1, 5):
        request.applymarker(pytest.mark.xfail)

    result = benchmark(q1, library)

    compare_dicts(result, q1_expected)

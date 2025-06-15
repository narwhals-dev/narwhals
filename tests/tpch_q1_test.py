from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING
from unittest import mock

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import DASK_VERSION, PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from narwhals.stable.v1.typing import IntoFrame


@pytest.mark.parametrize("library", ["pandas", "polars", "pyarrow", "dask"])
@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
def test_q1(library: str) -> None:
    if library == "dask" and DASK_VERSION < (2024, 10):
        pytest.skip()
    if library == "pandas" and PANDAS_VERSION < (1, 5):
        pytest.skip()
    elif library == "pandas":
        df_raw: IntoFrame = pd.read_csv("tests/data/lineitem.csv")
    elif library == "polars":
        pytest.importorskip("polars")
        import polars as pl

        df_raw = pl.scan_csv("tests/data/lineitem.csv")
    elif library == "dask":
        pytest.importorskip("dask")
        import dask.dataframe as dd

        df_raw = dd.from_pandas(
            pd.read_csv("tests/data/lineitem.csv", dtype_backend="pyarrow")
        )
    else:
        pytest.importorskip("pyarrow.csv")
        import pyarrow.csv as pa_csv

        df_raw = pa_csv.read_csv("tests/data/lineitem.csv")
    var_1 = datetime(1998, 9, 2)
    df = nw.from_native(df_raw).lazy()
    schema = df.collect_schema()
    if schema["l_shipdate"] != nw.Date and schema["l_shipdate"] != nw.Datetime:
        df = df.with_columns(nw.col("l_shipdate").str.to_datetime())
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
            ]
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    expected = {
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
    assert_equal_data(query_result, expected)


@pytest.mark.parametrize("library", ["pandas", "polars"])
@pytest.mark.filterwarnings(
    "ignore:.*Passing a BlockManager.*:DeprecationWarning",
    "ignore:.*Complex.*:UserWarning",
)
def test_q1_w_generic_funcs(library: str) -> None:
    if library == "pandas" and PANDAS_VERSION < (1, 5):
        pytest.skip()
    elif library == "pandas":
        df_raw: IntoFrame = pd.read_csv("tests/data/lineitem.csv")
    else:
        pytest.importorskip("polars")
        import polars as pl

        df_raw = pl.read_csv("tests/data/lineitem.csv")
    var_1 = datetime(1998, 9, 2)
    df = nw.from_native(df_raw, eager_only=True)
    df = df.with_columns(nw.col("l_shipdate").str.to_datetime())
    query_result = (
        df.filter(nw.col("l_shipdate") <= var_1)
        .with_columns(
            charge=(
                nw.col("l_extendedprice")
                * (1.0 - nw.col("l_discount"))
                * (1.0 + nw.col("l_tax"))
            )
        )
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            nw.sum("l_quantity").alias("sum_qty"),
            nw.sum("l_extendedprice").alias("sum_base_price"),
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            nw.col("charge").sum().alias("sum_charge"),
            nw.mean("l_quantity").alias("avg_qty"),
            nw.mean("l_extendedprice").alias("avg_price"),
            nw.mean("l_discount").alias("avg_disc"),
            count_order=nw.len(),
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    result = query_result.to_dict(as_series=False)
    expected = {
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
    assert_equal_data(result, expected)


@mock.patch.dict(os.environ, {"NARWHALS_FORCE_GENERIC": "1"})
@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
def test_q1_w_pandas_agg_generic_path() -> None:
    df_raw = pd.read_csv("tests/data/lineitem.csv")
    df_raw["l_shipdate"] = pd.to_datetime(df_raw["l_shipdate"])
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
                nw.sum("l_quantity").alias("sum_qty"),
                nw.sum("l_extendedprice").alias("sum_base_price"),
                nw.sum("disc_price").alias("sum_disc_price"),
                nw.col("charge").sum().alias("sum_charge"),
                nw.mean("l_quantity").alias("avg_qty"),
                nw.mean("l_extendedprice").alias("avg_price"),
                nw.mean("l_discount").alias("avg_disc"),
                nw.len().alias("count_order"),
            ]
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    expected = {
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
    assert_equal_data(query_result, expected)

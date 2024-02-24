from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from unittest import mock

import polars
import pytest

from narwhals import translate_frame
from tests.utils import compare_dicts


@pytest.mark.parametrize(
    "df_raw",
    [
        (polars.read_parquet("tests/data/lineitem.parquet").to_pandas()),
        polars.scan_parquet("tests/data/lineitem.parquet"),
    ],
)
def test_q1(df_raw: Any) -> None:
    var_1 = datetime(1998, 9, 2)
    df, pl = translate_frame(df_raw, lazy_only=True)
    query_result = (
        df.filter(pl.col("l_shipdate") <= var_1)
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                pl.sum("l_quantity").alias("sum_qty"),
                pl.sum("l_extendedprice").alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    pl.col("l_extendedprice")
                    * (1.0 - pl.col("l_discount"))
                    * (1.0 + pl.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                pl.mean("l_quantity").alias("avg_qty"),
                pl.mean("l_extendedprice").alias("avg_price"),
                pl.mean("l_discount").alias("avg_disc"),
                pl.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    result = query_result.collect().to_dict(as_series=False)
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
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    "df_raw",
    [
        (polars.read_parquet("tests/data/lineitem.parquet").to_pandas()),
    ],
)
@mock.patch.dict(os.environ, {"NARWHALS_FORCE_GENERIC": "1"})
def test_q1_w_pandas_agg_generic_path(df_raw: Any) -> None:
    var_1 = datetime(1998, 9, 2)
    df, pl = translate_frame(df_raw, lazy_only=True)
    query_result = (
        df.filter(pl.col("l_shipdate") <= var_1)
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                pl.sum("l_quantity").alias("sum_qty"),
                pl.sum("l_extendedprice").alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    pl.col("l_extendedprice")
                    * (1.0 - pl.col("l_discount"))
                    * (1.0 + pl.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                pl.mean("l_quantity").alias("avg_qty"),
                pl.mean("l_extendedprice").alias("avg_price"),
                pl.mean("l_discount").alias("avg_disc"),
                pl.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    result = query_result.collect().to_dict(as_series=False)
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
    compare_dicts(result, expected)

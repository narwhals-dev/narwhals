# ruff: noqa
import polars as pl
from typing import Any
from datetime import datetime
import narwhals as nw
import pandas as pd
import polars

polars.Config.set_tbl_cols(10)


def q1(df_raw: Any) -> Any:
    var_1 = datetime(1998, 9, 2)
    df = nw.from_native(df_raw)
    result = (
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
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    return nw.to_native(result)


df = pd.read_parquet(
    "../tpch-data/s1/lineitem.parquet", dtype_backend="pyarrow", engine="pyarrow"
)
print(q1(df))
df = pl.read_parquet("../tpch-data/s1/lineitem.parquet")
print(q1(df))
df = pl.scan_parquet("../tpch-data/s1/lineitem.parquet")
print(q1(df).collect())

# ruff: noqa
from typing import Any
from datetime import datetime
from narwhals import translate_frame, to_original_object
import pandas as pd
import polars

polars.Config.set_tbl_cols(10)


def q1(df_raw: Any) -> Any:
    var_1 = datetime(1998, 9, 2)
    df, pl = translate_frame(df_raw, lazy_only=True)
    result = (
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
    return result.collect().to_native()


# df = pd.read_parquet("../tpch-data/lineitem.parquet")
# df[["l_quantity", "l_extendedprice", "l_discount", "l_tax"]] = df[
#     ["l_quantity", "l_extendedprice", "l_discount", "l_tax"]
# ].astype("float64")
# df["l_shipdate"] = pd.to_datetime(df["l_shipdate"])
# print(q1(df))
df = polars.scan_parquet("../tpch-data/lineitem.parquet")
print(q1(df))

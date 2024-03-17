# ruff: noqa
from typing import Any
from datetime import datetime
import narwhals as nw
import pandas as pd
import polars

polars.Config.set_tbl_cols(10)


def q1(df_raw: Any) -> Any:
    var_1 = datetime(1998, 9, 2)
    df = nw.LazyFrame(df_raw)
    result = (
        df.filter(nw.col("l_shipdate") <= var_1)
        .group_by(["l_returnflag", "l_linestatus"])
        .agg(
            [
                nw.sum("l_quantity").alias("sum_qty"),
                nw.sum("l_extendedprice").alias("sum_base_price"),
                (nw.col("l_extendedprice") * (1 - nw.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (
                    nw.col("l_extendedprice")
                    * (1.0 - nw.col("l_discount"))
                    * (1.0 + nw.col("l_tax"))
                )
                .sum()
                .alias("sum_charge"),
                nw.mean("l_quantity").alias("avg_qty"),
                nw.mean("l_extendedprice").alias("avg_price"),
                nw.mean("l_discount").alias("avg_disc"),
                nw.len().alias("count_order"),
            ],
        )
        .sort(["l_returnflag", "l_linestatus"])
    )
    return nw.to_native(result.collect())


# df = pd.read_parquet("../tpch-data/lineitem.parquet")
# df[["l_quantity", "l_extendedprice", "l_discount", "l_tax"]] = df[
#     ["l_quantity", "l_extendedprice", "l_discount", "l_tax"]
# ].astype("float64")
# df["l_shipdate"] = pd.to_datetime(df["l_shipdate"])
# print(q1(df))
df = polars.scan_parquet("../tpch-data/lineitem.parquet")
print(q1(df))

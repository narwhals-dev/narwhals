import pandas as pd
import polars_api_compat
import polars as pl
from datetime import datetime

df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})

dfx, plx = polars_api_compat.convert(df, api_version='0.20')
print(dfx.dataframe)

var_1 = datetime(1998, 9, 2)
# q = pl.scan_parquet('../tpch-data/lineitem.parquet')

q = pl.scan_parquet('../tpch-data/lineitem.parquet').collect().to_pandas()
q['l_shipdate'] = pd.to_datetime(q['l_shipdate'])
qx, plx = polars_api_compat.convert(q, api_version='0.20')
qx, plx = pl.scan_parquet('../tpch-data/lineitem.parquet'), pl
var_1 = datetime(1998, 9, 2)
q_final = (
    qx.filter(plx.col("l_shipdate") <= var_1)
    .group_by(["l_returnflag", "l_linestatus"])
    .agg(
        plx.sum("l_quantity").alias("sum_qty"),
        # plx.sum("l_extendedprice").alias("sum_base_price"),
        # (plx.col("l_extendedprice") * (1 - plx.col("l_discount")))
        # .sum()
        # .alias("sum_disc_price"),
        # (
        #     plx.col("l_extendedprice")
        #     * (1.0 - plx.col("l_discount"))
        #     * (1.0 + plx.col("l_tax"))
        # )
        # .sum()
        # .alias("sum_charge"),
        # plx.mean("l_quantity").alias("avg_qty"),
        # plx.mean("l_extendedprice").alias("avg_price"),
        # plx.mean("l_discount").alias("avg_disc"),
        # plx.len().alias("count_order"),
    )
    .sort("l_returnflag", "l_linestatus")
)

print(q_final.collect())
# print(q_final.collect().dataframe)

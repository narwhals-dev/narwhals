from datetime import datetime
from polars_api_compat import to_polars_api, to_native
import polars


def q() -> None:
    var_1 = datetime(1998, 9, 2)
    q = polars.scan_parquet("../tpch-data/lineitem.parquet").collect().to_pandas()
    q, pl = to_polars_api(q, version="0.20")
    q_final = (
        q.filter(pl.col("l_shipdate") <= var_1)
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

    print(to_native(q_final.collect()))


if __name__ == "__main__":
    q()

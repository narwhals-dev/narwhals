# ruff: noqa
# type: ignore
import pandas as pd
import polars as pl

import narwhals as nw


def my_agnostic_function(
    suppliers_native,
    parts_native,
):
    suppliers = nw.LazyFrame(suppliers_native)
    parts = nw.LazyFrame(parts_native)

    result = (
        suppliers.join(parts, left_on="city", right_on="city")
        .filter(nw.col("weight") > 10)
        .group_by("s")
        .agg(
            weight_mean=nw.col("weight").mean(),
            weight_max=nw.col("weight").max(),
        )
    )
    return nw.to_native(result)


suppliers = {
    "s": ["S1", "S2", "S3", "S4", "S5"],
    "sname": ["Smith", "Jones", "Blake", "Clark", "Adams"],
    "status": [20, 10, 30, 20, 30],
    "city": ["London", "Paris", "Paris", "London", "Athens"],
}
parts = {
    "p": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "pname": ["Nut", "Bolt", "Screw", "Screw", "Cam", "Cog"],
    "color": ["Red", "Green", "Blue", "Red", "Blue", "Red"],
    "weight": [12.0, 17.0, 17.0, 14.0, 12.0, 19.0],
    "city": ["London", "Paris", "Oslo", "London", "Paris", "London"],
}

print("pandas output:")
print(
    my_agnostic_function(
        pd.DataFrame(suppliers),
        pd.DataFrame(parts),
    )
)
print("\nPolars output:")
print(
    my_agnostic_function(
        pl.LazyFrame(suppliers),
        pl.LazyFrame(parts),
    ).collect()
)

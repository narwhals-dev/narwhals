# ruff: noqa
from typing import TypeVar
import pandas as pd
import polars as pl

from puffin import to_polars_api, to_original_object

AnyDataFrame = TypeVar("AnyDataFrame")


def my_agnostic_function(
    suppliers_native: AnyDataFrame,
    parts_native: AnyDataFrame,
) -> AnyDataFrame:
    suppliers, pl = to_polars_api(suppliers_native, version="0.20")
    parts, _ = to_polars_api(parts_native, version="0.20")
    result = (
        suppliers.join(parts, left_on="city", right_on="city")
        .filter(
            pl.col("color").is_in(["Red", "Green"]),
            pl.col("weight") > 14,
        )
        .group_by("s", "p")
        .agg(
            weight_mean=pl.col("weight").mean(),
            weight_max=pl.col("weight").max(),
        )
    )
    return to_original_object(result.collect())


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
    )
)

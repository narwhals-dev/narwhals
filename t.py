from __future__ import annotations

import daft

import narwhals as nw
import polars as pl

# df = nw.from_native(daft.from_pydict({"a": [1, 2, 3]}))

# print(df)
# print(type(df))

daft_df = daft.from_pydict({
        "A": [1, 2, 3, 4, 5],
        "fruits": ["banana", "banana", "apple", "apple", "banana"],
        "B": [5, 4, 3, 2, 1],
    })

nw_l = nw.from_native(daft_df)

print(type(nw_l))

# turn this into a nw Dataframe
nw_df = nw_l.collect()

print(type(nw_df))

# now we can go to polars!
pl_df = nw_df.to_polars()

print(pl_df)

print(type(pl_df))

# ... and do a typical polars operation
print(pl_df.with_columns(a_max=pl.col("A").max().over(pl.col("fruits"))))




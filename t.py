# ruff: noqa
# type: ignore
import polars
import pandas as pd
import polars as pl

import narwhals as nw

df_raw = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df = nw.LazyFrame(df_raw)
df_raw_2 = pd.DataFrame({"a": [1, 3], "c": [7, 9]})
df2 = nw.LazyFrame(df_raw_2)

result = df.sort("a", "b")
print(nw.to_native(result))

result = df.filter(nw.col("a") > 1)
print(nw.to_native(result))

result = df.with_columns(
    c=nw.col("a") + nw.col("b"),
    d=nw.col("a") - nw.col("a").mean(),
)
print(nw.to_native(result))
result = df.with_columns(nw.all() * 2)
print(nw.to_native(result))

result = df.with_columns(horizonal_sum=nw.sum_horizontal(nw.col("a"), nw.col("b")))
print(nw.to_native(result))
result = df.with_columns(horizonal_sum=nw.sum_horizontal("a", nw.col("b")))
print(nw.to_native(result))


result = df.select(nw.all().sum())
print(nw.to_native(result))
result = df.select(nw.col("a", "b") * 2)
print(nw.to_native(result))

# # TODO!
# # result = (
# #     df.collect()
# #     .group_by("b")
# #     .agg(
# #         nw.all().sum(),
# #     )
# # )
# # print(nw.to_native(result))

result = (
    df.collect()
    .group_by("b")
    .agg(
        nw.col("a").sum(),
        simple=nw.col("a").sum(),
        complex=(nw.col("a") + 1).sum(),
        other=nw.sum("a"),
    )
)
print(nw.to_native(result))
print("multiple simple")
result = (
    df.collect()
    .group_by("b")
    .agg(
        nw.col("a", "z").sum(),
    )
)
print(nw.to_native(result))

result = df.join(df2, left_on="a", right_on="a")
print(nw.to_native(result))


result = df.rename({"a": "a_new", "b": "b_new"})
print(nw.to_native(result))

result = df.collect().to_dict()
print(result)
print(polars.from_pandas(nw.to_native(df)).to_dict())

result = df.collect().to_dict(as_series=False)
print("this")
print(result)
print("that")
print(polars.from_pandas(nw.to_native(df)).to_dict(as_series=False))

agg = (nw.col("b") - nw.col("z").mean()).mean()
print(nw.to_native(df.with_columns(d=agg)))
result = df.group_by("a").agg(agg)
print(nw.to_native(result))

print(nw.col("a") + nw.col("b"))
print(nw.col("a", "b").sum())

result = df.select(nw.col("a", "b").sum())
print(nw.to_native(result))

print(df.schema)
print(df.schema["a"].is_numeric())

df_raw = pd.DataFrame(
    {
        "a": [1, 3, 2],
        "b": [4.0, 4, 6],
        "c": ["a", "b", "c"],
        "d": [True, False, True],
    }
)
df = nw.DataFrame(df_raw)
print(df.schema)
print(df.schema["a"].is_numeric())
print(df.schema["b"].is_numeric())
print(df.schema["c"].is_numeric())
print(df.schema["d"].is_numeric())

result = df.with_columns(nw.col("a").cast(nw.Float32))
print(nw.to_native(result))
print(result._dataframe._dataframe.dtypes)

print(df.schema)
result = df.select([col for (col, dtype) in df.schema.items() if dtype == nw.Float64])
print(nw.to_native(result))
print(result._dataframe._dataframe.dtypes)

result = df.select("a", "b").select(nw.all() + nw.col("a"))
print(nw.to_native(result))

df = nw.DataFrame(df_raw, features=["eager"])
print(df["a"].mean())
df = nw.DataFrame(pl.from_pandas(df_raw), features=["eager"])
print(df["a"].mean())

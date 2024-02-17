import pandas as pd
import polars_api_compat

df_raw = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6]})
df, pl = polars_api_compat.to_polars_api(df_raw, version="0.20")
df_raw_2 = pd.DataFrame({"a": [1, 3], "c": [7, 9]})
df2, pl = polars_api_compat.to_polars_api(df_raw_2, version="0.20")

result = df.sort("a", "b")
print(result.dataframe)

result = df.filter(pl.col("a") > 1)
print(result.dataframe)

result = df.with_columns(
    c=pl.col("a") + pl.col("b"),
    d=pl.col("a") - pl.col("a").mean(),
)
print(result.dataframe)
result = df.with_columns(pl.all() * 2)
print(result.dataframe)

result = df.with_columns(horizonal_sum=pl.sum_horizontal(pl.col("a"), pl.col("b")))
print(result.dataframe)
result = df.with_columns(horizonal_sum=pl.sum_horizontal("a", pl.col("b")))
print(result.dataframe)


result = df.select(pl.all().sum())
print(result.dataframe)
result = df.select(pl.col("a", "b") * 2)
print(result.dataframe)


result = (
    df.collect()
    .group_by("b")
    .agg(
        pl.col("a").sum(),
        simple=pl.col("a").sum(),
        complex=(pl.col("a") + 1).sum(),
        other=pl.sum("a"),
    )
)
print(result.dataframe)

result = df.join(df2, left_on="a", right_on="a")
print(result.dataframe)

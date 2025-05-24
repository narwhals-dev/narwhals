# Empty aggregations

What is the sum of zero values? As it turns out, tools disagree:

```python exec="1" result="python" session="empty aggregations" source="above"
import narwhals as nw
import polars as pl
import duckdb

polars_df = pl.DataFrame({"a": [None], "b": [1]}, schema={"a": pl.Int64, "b": pl.Int64})
print("Polars result")
print(polars_df.group_by("b").agg(pl.col("a").sum()))

print("DuckDB result")
print(duckdb.sql("""from polars_df select b, sum(a) as a group by b"""))
```

Polars, pandas, and PyArrow think the result is zero. SQL engines think it's `NULL`. Who's correct?

For now, we respect each backend's opinion and leave this result backend-specific, to avoid
interfering with how aggregations compose with other operations. If it's crucial to you
that an empty sum returns `0` for all backends, you can always follow the sum with
`fill_null(0)`.

```python exec="1" result="python" session="empty aggregations" source="above"
from narwhals.typing import IntoFrameT


def custom_group_by_sum(df_native: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(df_native)
        .group_by("b")
        .agg(nw.col("a").sum())
        .with_columns(nw.col("a").fill_null(0))
    )


print("Polars result:")
print(custom_group_by_sum(polars_df))
print("DuckDB result:")
print(custom_group_by_sum(duckdb.table("polars_df")))
```

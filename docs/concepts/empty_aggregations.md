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
interfering with how aggregations compose with other operations.

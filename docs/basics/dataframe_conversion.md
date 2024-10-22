# Lightweight DataFrame conversion

Some library maintainers must apply complex data frame operations, using methods and functions that may not (yet) be implemented in Narwhals. In such cases, Narwhals can still be highly beneficial, by allowing easy data frame conversion.

Imagine that you maintain a library with a function that operates on Pandas data frames to produce automated reports. You want to allow users to supply data frames in any format to that function (Pandas, Polars, DuckDB, CuDF, Modin, etc.) without adding all those dependencies to your own project.

One solution is to use Narwhals as a thin "DataFrame ingestion" layer, to convert user-supplied data frames to the format that your library uses internally. Since Narwhals is 0-dependency, this is a much more lightweight solution than including all the data frame libraries as dependencies.

For example, this function can ingest any data frame type supported by Narwhals, and convert it to a Pandas DataFrame for internal use:

```python
import narwhals as nw
def df_to_pandas(df):
  out = nw.from_native(df)
  out = out.to_pandas()
  return out
```

Similarly, if your library uses Polars internally, you can convert any user-supplied data frames to Polars format using Narwhals.

```python
import narwhals as nw
import polars as pl
def df_to_polars(df):
  out = nw.from_native(df)  
  out = out.to_arrow()
  return pl.DataFrame(out)
```


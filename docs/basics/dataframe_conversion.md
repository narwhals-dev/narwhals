# Lightweight DataFrame conversion

Some library maintainers must apply complex data frame operations, using methods and functions that may not (yet) be implemented in Narwhals. In such cases, Narwhals can still be highly beneficial, by allowing easy data frame conversion.

Imagine that you maintain a library with a function that operates on Pandas data frames to produce automated reports. You want to allow users to supply data frames in any format to that function (Pandas, Polars, DuckDB, CuDF, Modin, etc.) without adding all those dependencies to your own project.

One solution is to use Narwhals as a thin "DataFrame ingestion" layer, to convert user-supplied data frames to the format that your library uses internally. Since Narwhals is 0-dependency, this is a much more lightweight solution than including all the data frame libraries as dependencies.

To illustrate, we create data frames in various formats:

```python exec="1" source="above"
import narwhals as nw
from narwhals.typing import IntoDataFrame

import duckdb
import polars as pl
import pandas as pd

df_polars = pl.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "fruits": ["banana", "banana", "apple", "apple", "banana"],
        "B": [5, 4, 3, 2, 1],
        "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
    }
)
df_pandas = df_polars.to_pandas()
df_duckdb = duckdb.sql("SELECT * FROM df_polars")
```

Now, we define a function that can ingest any data frame type supported by Narwhals, and convert it to a Pandas DataFrame for internal use:

```python exec="1" source="above"
def df_to_pandas(df: IntoDataFrame) -> pd.DataFrame:
    return nw.from_native(df).to_pandas()

df_to_pandas(df_polars)
```

Similarly, if your library uses Polars internally, you can convert any user-supplied data frame to Polars format using Narwhals.

```python exec="1" source="above"
def df_to_polars(df: IntoDataFrame) -> pl.DataFrame:
    return nw.from_arrow(nw.from_native(df), native_namespace=pl).to_native()

df_to_polars(df_duckdb)
```

Note that the `df_to_polars` function defined above uses PyCapsule. This strategy does not guarantee that we can call the Arrow stream repeatedly. If you need to ingest the same data frame several times, you may want to go through PyArrow, which may be less efficient, but is more forgiving:

```python exec="1" source="above"
def df_to_polars(df: IntoDataFrame) -> pl.DataFrame:
    return pl.DataFrame(nw.from_native(df).to_arrow())

df_duckdb = duckdb.sql("SELECT * FROM df_polars")
df_to_polars(df_duckdb)
```


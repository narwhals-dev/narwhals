# Conversion between libraries

Some library maintainers must apply complex dataframe operations, using methods and functions that may not (yet) be implemented in Narwhals. In such cases, Narwhals can still be highly beneficial, by allowing easy dataframe conversion.

## Dataframe X in, pandas out

Imagine that you maintain a library with a function that operates on pandas dataframes to produce automated reports. You want to allow users to supply a dataframe in any format to that function (pandas, Polars, DuckDB, cuDF, Modin, etc.) without adding all those dependencies to your own project and without special-casing each input library's variation of `to_pandas` / `toPandas` / `to_pandas_df` / `df` ...

One solution is to use Narwhals as a thin Dataframe ingestion layer, to convert user-supplied dataframe to the format that your library uses internally. Since Narwhals is zero-dependency, this is a much more lightweight solution than including all the dataframe libraries as dependencies,
and easier to write than special casing each input library's `to_pandas` method (if it even exists!).

To illustrate, we create dataframes in various formats:

```python exec="1" source="above" session="conversion"
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

Now, we define a function that can ingest any dataframe type supported by Narwhals, and convert it to a pandas DataFrame for internal use:

```python exec="1" source="above" session="conversion" result="python"
def df_to_pandas(df: IntoDataFrame) -> pd.DataFrame:
    return nw.from_native(df).to_pandas()


print(df_to_pandas(df_polars))
```

## Dataframe X in, Polars out

### Via PyCapsule Interface

Similarly, if your library uses Polars internally, you can convert any user-supplied dataframe to Polars format using Narwhals.

```python exec="1" source="above" session="conversion" result="python"
def df_to_polars(df: IntoDataFrame) -> pl.DataFrame:
    return nw.from_arrow(nw.from_native(df), native_namespace=pl).native


print(df_to_polars(df_duckdb))  # You can only execute this line of code once.
```

It works to pass Polars to `native_namespace` here because Polars supports the [PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) for import.

Note that the PyCapsule Interface makes no guarantee that you can call it repeatedly, so the approach above only works if you
only expect to perform the conversion a single time on each input object.

### Via PyArrow

If you need to ingest the same dataframe multiple times, then you may want to go via PyArrow instead.
This may be less efficient than the PyCapsule approach above (and always requires PyArrow!), but is more forgiving:

```python exec="1" source="above" session="conversion" result="python"
def df_to_polars(df: IntoDataFrame) -> pl.DataFrame:
    return pl.DataFrame(nw.from_native(df).to_arrow())


df_duckdb = duckdb.sql("SELECT * FROM df_polars")
print(df_to_polars(df_duckdb))  # We can execute this...
print(df_to_polars(df_duckdb))  # ...as many times as we like!
```

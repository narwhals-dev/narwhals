# Boolean columns

Generally speaking, Narwhals operations preserve null values.
For example, if you do `nw.col('a')*2`, then:

- Values which were non-null get multiplied by 2.
- Null values stay null.

```python exec="1" source="above" session="boolean" result="python"
import narwhals as nw

import pandas as pd
import polars as pl
import pyarrow as pa

data = {"a": [1.4, None, 4.2]}
print("pandas output")
print(nw.from_native(pd.DataFrame(data)).with_columns(b=nw.col("a") * 2).to_native())
print("\nPolars output")
print(nw.from_native(pl.DataFrame(data)).with_columns(b=nw.col("a") * 2).to_native())
print("\nPolars output")
print(nw.from_native(pa.table(data)).with_columns(b=nw.col("a") * 2).to_native())
```

What do we do, however, when the result column is boolean? For
example, `nw.col('a') > 0`?
Unfortunately, this is backend-dependent:

- for all backends except pandas, null values are preserved
- for pandas, this depends on the dtype backend:
    - for PyArrow dtypes and pandas nullable dtypes, null
      values are preserved
    - for the classic NumPy dtypes, null values are typically
      filled in with `False`.

pandas is generally moving towards nullable dtypes, and they
may become the default in the future, so we hope that the
classical NumPy dtypes not supporting null values will just
be a temporary legacy pandas issue which will eventually go
away anyway.

```python exec="1" source="above" session="boolean" result="python"
print("pandas output")
print(nw.from_native(pd.DataFrame(data)).with_columns(b=nw.col("a") > 2).to_native())
print("\npandas (nullable dtypes) output")
print(
    nw.from_native(pd.DataFrame(data, dtype="Float64"))
    .with_columns(b=nw.col("a") > 2)
    .to_native()
)
print("\npandas (pyarrow dtypes) output")
print(
    nw.from_native(pd.DataFrame(data, dtype="Float64[pyarrow]"))
    .with_columns(b=nw.col("a") > 2)
    .to_native()
)
print("\nPolars output")
print(nw.from_native(pl.DataFrame(data)).with_columns(b=nw.col("a") > 2).to_native())
print("\nPyArrow output")
print(nw.from_native(pa.table(data)).with_columns(b=nw.col("a") > 2).to_native())
```
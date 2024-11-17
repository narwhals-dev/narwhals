# Boolean columns

Generally speaking, Narwhals operations preserve null values.
For example, if you do `nw.col('a')*2`, then:

- Values which were non-null get multiplied by 2.
- Null values stay null.

```python exec="1" source="above" session="boolean"
import narwhals as nw
from narwhals.typing import FrameT

data = {"a": [1.4, None, 4.2]}


def multiplication(df: FrameT) -> FrameT:
    return nw.from_native(df).with_columns((nw.col("a") * 2).alias("a*2")).to_native()
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="boolean"
    import pandas as pd

    df = pd.DataFrame(data)
    print(multiplication(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="boolean"
    import polars as pl

    df = pl.DataFrame(data)
    print(multiplication(df))
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="boolean"
    import pyarrow as pa

    table = pa.table(data)
    print(multiplication(table))
    ```

What do we do, however, when the result column is boolean? For
example, `nw.col('a') > 0`?
Unfortunately, this is backend-dependent:

- for all backends except pandas, null values are preserved
- for pandas, this depends on the dtype backend:
    - for PyArrow dtypes and pandas nullable dtypes, null values are preserved
    - for the classic NumPy dtypes, null values are typically filled in with `False`.

pandas is generally moving towards nullable dtypes, and they
[may become the default in the future](https://github.com/pandas-dev/pandas/pull/58988),
so we hope that the classical NumPy dtypes not supporting null values will just
be a temporary legacy pandas issue which will eventually go
away anyway.

```python exec="1" source="above" session="boolean"
def comparison(df: FrameT) -> FrameT:
    return nw.from_native(df).with_columns((nw.col("a") > 2).alias("a>2")).to_native()
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="boolean"
    import pandas as pd

    df = pd.DataFrame(data)
    print(comparison(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="boolean"
    import polars as pl

    df = pl.DataFrame(data)
    print(comparison(df))
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="boolean"
    import pyarrow as pa

    table = pa.table(data)
    print(comparison(table))
    ```

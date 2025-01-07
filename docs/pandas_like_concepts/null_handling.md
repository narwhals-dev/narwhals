# Null/NaN handling

pandas doesn't distinguish between Null and NaN values as Polars and PyArrow do.

Depending on the data type of the underlying data structure, `np.nan`, `pd.NaT`, `None` and `pd.NA` all encode missing data in pandas.

Polars and PyArrow, instead, treat `NaN` as a valid floating point value which is rare to encounter and more often produced as the result of a computation than explicitly set during data initialization; they treat `null` as the missing data indicator, regardless of the data type.

In Narwhals, then, `is_null` behaves differently across backends (and so do `drop_nulls`, `fill_null` and `null_count`):

```python exec="1" source="above" session="null_handling"
import narwhals as nw
import numpy as np
from narwhals.typing import IntoFrameT

data = {"a": [1.4, float("nan"), np.nan, 4.2, None]}


def check_null_behavior(df: IntoFrameT) -> IntoFrameT:
    return nw.from_native(df).with_columns(a_is_null=nw.col("a").is_null()).to_native()
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pandas as pd

    df = pd.DataFrame(data)
    print(check_null_behavior(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import polars as pl

    df = pl.DataFrame(data)
    print(check_null_behavior(df))
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pyarrow as pa

    df = pa.table(data)
    print(check_null_behavior(df))
    ```

Conversely, `is_nan` is consistent across backends. This consistency comes from Narwhals exploiting its native implementations
in Polars and PyArrow, while ensuring that pandas only identifies the floating-point NaN values and not those encoding the missing value indicator.

```python exec="1" source="above" session="null_handling"
import narwhals as nw
from narwhals.typing import IntoFrameT

data = {"a": [0.0, None, 2.0]}


def check_nan_behavior(df: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(df)
        .with_columns(
            a_div_a=(nw.col("a") / nw.col("a")),
            a_div_a_is_nan=(nw.col("a") / nw.col("a")).is_nan(),
        )
        .to_native()
    )
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pandas as pd

    df = pd.DataFrame(data).astype({"a": "Float64"})
    print(check_nan_behavior(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import polars as pl

    df = pl.DataFrame(data)
    print(check_nan_behavior(df))
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pyarrow as pa

    df = pa.table(data)
    print(check_nan_behavior(df))
    ```
